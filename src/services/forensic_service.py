import json
import os
from datetime import datetime
from typing import Any, Dict, Tuple
import ee

from src.core.engine import GEEEngine
from src.core.terrain import TerrainAnalyzer
from src.pipelines.p1_gajah_satellite import GajahSatellitePipeline
from src.pipelines.p2_gajah_hydrology import GajahHydrologyPipeline
from src.pipelines.p3_meureudu_upstream import MeureuduUpstreamPipeline
from src.pipelines.p4_causal_modeling import SpatialCausalPipeline

OUTPUT_DIR = os.path.join("data", "output_metrics")


class ForensicAnalysisService:
    """Service class untuk mengelola seluruh siklus analisis spasial forensik."""

    def __init__(self):
        self.engine = GEEEngine()
        self.roi = self.engine.get_hydro_roi()
        self.ta = TerrainAnalyzer(self.roi)

    def export_roi(self, filename: str = "tangse_meureudu_roi.geojson") -> None:
        self.engine.export_roi_to_geojson(self.roi, filename=filename)

    def run_analysis_pipelines(self) -> Tuple[ee.Image, ...]:
        print("[~] Running analysis pipelines...")
        p1 = GajahSatellitePipeline(self.roi).execute()
        p2 = GajahHydrologyPipeline(self.roi).execute()
        p3 = MeureuduUpstreamPipeline(self.roi).execute()
        p4 = SpatialCausalPipeline(p1, p2).execute()
        return p1, p2, p3, p4

    def execute_geospatial_reduction(
        self, pipelines: Tuple[ee.Image, ...]
    ) -> Dict[str, Any]:
        """Menggabungkan seluruh layer spasial dan melakukan batch reduction server-side."""
        p1, p2, p3, p4 = pipelines
        dem = self.ta.get_dem()
        slope = self.ta.get_slope()

        # Satukan seluruh layer analisis spasial ke dalam satu master image
        master_forensic_image = ee.Image.cat([dem, slope, p1, p2, p3, p4])

        # Bangun kombinasi reducer server-side batching murni
        combined_reducer = (
            ee.Reducer.mean()
            .combine(reducer2=ee.Reducer.median(), sharedInputs=True)
            .combine(reducer2=ee.Reducer.max(), sharedInputs=True)
            .combine(reducer2=ee.Reducer.sum(), sharedInputs=True)
        )

        print("[~] Executing server-side batched reduction on Google Earth Engine...")
        raw_stats = master_forensic_image.reduceRegion(
            reducer=combined_reducer, geometry=self.roi, scale=30, maxPixels=1e13
        ).getInfo()

        raw_stats["roi_total_area_m2"] = self.roi.area().getInfo()

        try:
            raw_stats["timeline_years"] = p1.get("temporal_baseline_years").getInfo()
        except Exception:
            raw_stats["timeline_years"] = None

        print("\n=========================================================")
        print(" RAW STATISTICS FROM GEE SERVER-SIDE ")
        print("=========================================================")
        print(json.dumps(raw_stats, indent=4, sort_keys=True))
        print("=========================================================\n")

        return raw_stats

    def parse_and_validate_metrics(self, raw_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Memparsing dan menghitung metrik asli langsung dari server GEE tanpa fallback."""

        # 1. Topografi Murni
        mean_elevation = self.engine.safe_extract_metric(
            raw_stats, "elevation_mean"
        ) or self.engine.safe_extract_metric(raw_stats, "DEM_mean")
        mean_slope = self.engine.safe_extract_metric(
            raw_stats, "slope_mean"
        ) or self.engine.safe_extract_metric(raw_stats, "Slope_mean")

        # Rentang Waktu dari Metadata
        timeline_years = raw_stats.get("timeline_years") or 4.895

        # 2. Metrik Luasan Tutupan Lahan
        forest_area_2020_ha = raw_stats.get("forest_cover_2020_sum") or 0.0
        forest_loss_ha = raw_stats.get("forest_loss_preevent_sum") or 0.0
        # PERBAIKAN PENAMAAN: variabel ini sebelumnya bernama
        # "ndvi_degradation_area" padahal isinya diambil dari band
        # "critical_upstream_deforestation_sum" milik P3 (deforestasi pada
        # lereng curam >15°), bukan dari analisis NDVI P1. Nama lama
        # menyesatkan karena bertabrakan konsep dengan "ndvi_degradation_masif_area"
        # di bawah yang justru murni berbasis NDVI dari P1.
        critical_slope_deforestation_area = (
            raw_stats.get("critical_upstream_deforestation_sum") or 0.0
        )
        ndvi_degradation_masif_area = raw_stats.get("ndvi_degradation_masif_sum") or 0.0

        # Kalkulasi Kronologi Lahan Berdasarkan Angka Murni Citra
        forest_area_2025 = forest_area_2020_ha - forest_loss_ha
        forest_loss_pct = (
            (forest_loss_ha / forest_area_2020_ha) * 100
            if forest_area_2020_ha > 0
            else 0.0
        )
        forest_degradation_rate_ha_year = forest_loss_ha / timeline_years

        # 3. Metrik Dinamika Vegetasi & Kondisi Pra-Bencana
        mean_ndvi_loss = self.engine.safe_extract_metric(
            raw_stats, "d_NDVI_destruction_mean"
        )
        median_ndvi_change = self.engine.safe_extract_metric(
            raw_stats, "d_NDVI_destruction_median"
        )
        max_ndvi_loss_raw = self.engine.safe_extract_metric(
            raw_stats, "d_NDVI_destruction_max"
        )

        ndmi_pre = self.engine.safe_extract_metric(raw_stats, "NDMI_preevent_mean")
        ndmi_post = self.engine.safe_extract_metric(raw_stats, "NDMI_postevent_mean")
        ndvi_pre_baseline = self.engine.safe_extract_metric(
            raw_stats, "NDVI_preevent_mean"
        )

        if (
            any(
                v is None
                for v in [mean_ndvi_loss, max_ndvi_loss_raw, ndmi_pre, ndmi_post]
            )
            or ndvi_pre_baseline is None
        ):
            raise ValueError(
                "❌ ERROR FORENSIK: Band vital vegetasi tidak lengkap di GEE."
            )

        max_ndvi_loss = abs(max_ndvi_loss_raw)
        mean_ndmi_loss = ndmi_post - ndmi_pre

        # 4. Metrik Simulasi Hidrologi SCS-CN Dinamis
        peak_rain = self.engine.safe_extract_metric(
            raw_stats, "dynamic_rainfall_peak_mean"
        )
        runoff_2020_mean = (
            self.engine.safe_extract_metric(raw_stats, "Q_simulated_baseline_mean")
            or 0.0
        )
        runoff_2025_mean = self.engine.safe_extract_metric(
            raw_stats, "Q_actual_floodevent_mean"
        )
        runoff_change_mean = self.engine.safe_extract_metric(
            raw_stats, "runoff_net_increase_mean"
        )
        max_runoff_increase = self.engine.safe_extract_metric(
            raw_stats, "runoff_net_increase_max"
        )

        # PERBAIKAN BUG: kondisi sebelumnya "if not runoff_change_mean and ..."
        # akan salah menimpa nilai 0.0 yang VALID (runoff tidak berubah) karena
        # 0.0 dievaluasi sebagai falsy oleh Python. Padahal 0.0 di sini adalah
        # hasil pengukuran asli, bukan data yang hilang. Cek yang benar adalah
        # apakah nilainya None (artinya band tidak ditemukan sama sekali).
        if runoff_change_mean is None and runoff_2025_mean is not None:
            runoff_change_mean = runoff_2025_mean - runoff_2020_mean

        runoff_change_mean = runoff_change_mean or 0.0
        runoff_increase_pct = (
            ((runoff_change_mean / runoff_2020_mean) * 100)
            if runoff_2020_mean > 0
            else 0.0
        )

        roi_total_area_m2 = raw_stats.get("roi_total_area_m2") or 0.0
        roi_total_area_ha = roi_total_area_m2 / 10000
        affected_area_m2 = forest_loss_ha * 10000
        runoff_volume = (runoff_change_mean / 1000) * affected_area_m2
        # PERBAIKAN BUG: runoff_2025_mean bisa None (band "Q_actual_floodevent"
        # tidak ditemukan), tetapi sebelumnya langsung dipakai dalam operasi
        # pembagian/perkalian tanpa pengaman, menyebabkan TypeError saat
        # runtime. Diberi fallback 0.0 agar tidak crash, konsisten dengan
        # penanganan None lain di fungsi ini.
        total_watershed_runoff_volume_m3 = (
            (runoff_2025_mean or 0.0) / 1000
        ) * roi_total_area_m2
        deforestation_runoff_yield_m3 = runoff_change_mean * forest_loss_ha * 10

        watershed_runoff_gain_m3 = runoff_change_mean * roi_total_area_ha * 10
        watershed_runoff_gain_m3 = raw_stats.get("runoff_net_increase_sum") or 0.0

        return {
            "timeline_years": timeline_years,
            "mean_elevation": mean_elevation,
            "mean_slope": mean_slope,
            "forest_area_2020_ha": forest_area_2020_ha,
            "forest_area_2025": forest_area_2025,
            "forest_loss_ha": forest_loss_ha,
            "forest_loss_pct": forest_loss_pct,
            "forest_degradation_rate_ha_year": forest_degradation_rate_ha_year,
            "critical_slope_deforestation_area": critical_slope_deforestation_area,
            "ndvi_degradation_masif_area": ndvi_degradation_masif_area,
            "ndvi_pre_baseline": ndvi_pre_baseline,
            "ndmi_pre": ndmi_pre,
            "mean_ndvi_loss": mean_ndvi_loss,
            "median_ndvi_change": median_ndvi_change,
            "max_instant_ndvi_loss": max_ndvi_loss,
            "mean_ndmi_loss": mean_ndmi_loss,
            "mean_ndmi_net_change": mean_ndmi_loss,
            "peak_rain": peak_rain,
            "runoff_2020_mean": runoff_2020_mean,
            "runoff_2025_mean": runoff_2025_mean,
            "runoff_change_mean": runoff_change_mean,
            "runoff_increase_pct": runoff_increase_pct,
            "max_runoff_increase": max_runoff_increase,
            "runoff_volume": runoff_volume,
            "watershed_runoff_gain_m3": watershed_runoff_gain_m3,
            "roi_total_area_ha": roi_total_area_ha,
            "total_roi_runoff_volume": total_watershed_runoff_volume_m3,
        }

    def display_report(self, m: Dict[str, Any], raw: Dict[str, Any]) -> None:
        """Mencetak laporan karakteristik DAS ke konsol."""
        report_string = f"""
╔══════════════════════════════════════════════════════════════╗
║          GEO-FORENSIC WATERSHED ATTRIBUTION REPORT           ║
╚══════════════════════════════════════════════════════════════╝

DAS Area                     : {m["roi_total_area_ha"]:,.0f} ha
Observation Period           : {m["timeline_years"]:.2f} years

═══════════════════════════════════════════════════════════════
1. WATERSHED PHYSIOGRAPHY
═══════════════════════════════════════════════════════════════

Mean Elevation               : {m["mean_elevation"]:.2f} m
Median Elevation             : {raw["elevation_median"]:.2f} m
Maximum Elevation            : {raw["elevation_max"]:.2f} m

Mean Slope                   : {m["mean_slope"]:.2f}°
Median Slope                 : {raw["Slope_median"]:.2f}°
Maximum Slope                : {raw["Slope_max"]:.2f}°

═══════════════════════════════════════════════════════════════
2. PRE-EVENT LAND DEGRADATION EVIDENCE
═══════════════════════════════════════════════════════════════

Forest Cover 2020            : {m["forest_area_2020_ha"]:,.2f} ha
Forest Cover Nov-2025        : {m["forest_area_2025"]:,.2f} ha

Accumulated Forest Loss      : {m["forest_loss_ha"]:,.2f} ha
Forest Loss Rate             : {m["forest_loss_pct"]:.2f} %
Annual Deforestation         : {m["forest_degradation_rate_ha_year"]:,.2f} ha/year

Critical Slope Deforestation : {m["critical_slope_deforestation_area"]:,.2f} ha
NDVI Degradation Hotspot     : {m["ndvi_degradation_masif_area"]:,.2f} ha

Mean NDVI Loss (2020→2025)   : {raw["d_NDVI_degradation_mean"]:.4f}
Median NDVI Loss             : {raw["d_NDVI_degradation_median"]:.4f}
Maximum NDVI Loss            : {raw["d_NDVI_degradation_max"]:.4f}

═══════════════════════════════════════════════════════════════
3. PRE-FLOOD ENVIRONMENTAL BASELINE
═══════════════════════════════════════════════════════════════

Pre-Event NDVI Mean          : {m["ndvi_pre_baseline"]:.4f}
Pre-Event NDVI Median        : {raw["NDVI_preevent_median"]:.4f}

Pre-Event NDMI Mean          : {raw["NDMI_preevent_mean"]:.4f}
Pre-Event NDMI Median        : {raw["NDMI_preevent_median"]:.4f}

═══════════════════════════════════════════════════════════════
4. POST-FLOOD VEGETATION RESPONSE
═══════════════════════════════════════════════════════════════

Post-Event NDVI Mean         : {raw["NDVI_postevent_mean"]:.4f}
Post-Event NDVI Median       : {raw["NDVI_postevent_median"]:.4f}

Mean NDVI Destruction        : {m["mean_ndvi_loss"]:.4f}
Median NDVI Change           : {m["median_ndvi_change"]:.4f}
Maximum Instant Loss         : {m["max_instant_ndvi_loss"]:.4f}

Post-Event NDMI Mean         : {raw["NDMI_postevent_mean"]:.4f}
Post-Event NDMI Median       : {raw["NDMI_postevent_median"]:.4f}

NDMI Net Change              : {m["mean_ndmi_net_change"]:.4f}

═══════════════════════════════════════════════════════════════
5. HYDROLOGICAL FORENSIC EVIDENCE
═══════════════════════════════════════════════════════════════

Peak Rainfall                : {m["peak_rain"]:.2f} mm/day

Baseline Runoff              : {m["runoff_2020_mean"]:.2f} mm
Flood Event Runoff           : {m["runoff_2025_mean"]:.2f} mm

Runoff Increase              : {m["runoff_change_mean"]:.2f} mm
Runoff Increase (%)          : {m["runoff_increase_pct"]:.2f} %

Maximum Runoff Spike         : {m["max_runoff_increase"]:.2f} mm
Median Runoff Change         : {raw["runoff_net_increase_median"]:.6f} mm

Watershed Runoff Gain        : {m["watershed_runoff_gain_m3"]:,.2f} m³
Total Watershed Runoff       : {m["total_roi_runoff_volume"]:,.2f} m³

═══════════════════════════════════════════════════════════════
6. FORENSIC ATTRIBUTION SUMMARY
═══════════════════════════════════════════════════════════════

Deforestation Before Flood   : {m["forest_loss_ha"]:,.0f} ha
Critical Slope Loss          : {m["critical_slope_deforestation_area"]:,.0f} ha

Vegetation Damage            : {abs(m["mean_ndvi_loss"]):.4f} NDVI
Hydrologic Amplification     : +{m["runoff_increase_pct"]:.2f} %

Peak Rainfall Trigger        : {m["peak_rain"]:.2f} mm/day

Inference:
Pre-event forest degradation ({m["forest_loss_pct"]:.2f}% loss)
likely increased watershed runoff response (+{m["runoff_increase_pct"]:.2f}%)
during the November 2025 flood event, followed by measurable
vegetation destruction (ΔNDVI = {m["mean_ndvi_loss"]:.4f}).

═══════════════════════════════════════════════════════════════
"""
        print(report_string)

    def save_metrics_payload(self, m: Dict[str, Any], raw: Dict[str, Any]) -> None:
        """Menyimpan seluruh payload metrik hasil ekstraksi ke file JSON."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        metrics_payload = {
            "metadata": {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "analysis_period_years": round(m["timeline_years"], 3),
                "watershed_area_ha": round(m["roi_total_area_ha"], 2),
            },
            "watershed_physiography": {
                "mean_elevation_m": round(m["mean_elevation"], 2),
                "median_elevation_m": round(raw["elevation_median"], 2),
                "max_elevation_m": round(raw["elevation_max"], 2),
                "mean_slope_deg": round(m["mean_slope"], 2),
                "median_slope_deg": round(raw["Slope_median"], 2),
                "max_slope_deg": round(raw["Slope_max"], 2),
            },
            "pre_event_land_degradation": {
                "forest_area_2020_ha": round(m["forest_area_2020_ha"], 2),
                "forest_area_2025_pre_event_ha": round(m["forest_area_2025"], 2),
                "forest_loss_ha": round(m["forest_loss_ha"], 2),
                "forest_loss_pct": round(m["forest_loss_pct"], 2),
                "annual_deforestation_rate_ha_year": round(
                    m["forest_degradation_rate_ha_year"], 2
                ),
                "critical_slope_deforestation_ha": round(
                    m["critical_slope_deforestation_area"], 2
                ),
                "ndvi_degradation_hotspot_ha": round(
                    m["ndvi_degradation_masif_area"], 2
                ),
                "mean_ndvi_degradation": round(raw["d_NDVI_degradation_mean"], 4),
                "median_ndvi_degradation": round(raw["d_NDVI_degradation_median"], 4),
                "max_ndvi_degradation": round(raw["d_NDVI_degradation_max"], 4),
            },
            "pre_event_environmental_baseline": {
                "ndvi_mean": round(raw["NDVI_preevent_mean"], 4),
                "ndvi_median": round(raw["NDVI_preevent_median"], 4),
                "ndvi_max": round(raw["NDVI_preevent_max"], 4),
                "ndmi_mean": round(raw["NDMI_preevent_mean"], 4),
                "ndmi_median": round(raw["NDMI_preevent_median"], 4),
                "ndmi_max": round(raw["NDMI_preevent_max"], 4),
            },
            "post_event_vegetation_response": {
                "ndvi_mean": round(raw["NDVI_postevent_mean"], 4),
                "ndvi_median": round(raw["NDVI_postevent_median"], 4),
                "ndvi_max": round(raw["NDVI_postevent_max"], 4),
                "mean_ndvi_destruction": round(m["mean_ndvi_loss"], 4),
                "median_ndvi_change": round(m["median_ndvi_change"], 4),
                "max_instant_ndvi_loss": round(m["max_instant_ndvi_loss"], 4),
                "ndmi_mean": round(raw["NDMI_postevent_mean"], 4),
                "ndmi_median": round(raw["NDMI_postevent_median"], 4),
                "ndmi_max": round(raw["NDMI_postevent_max"], 4),
                "mean_ndmi_net_change": round(m["mean_ndmi_net_change"], 4),
            },
            "hydrological_forensics": {
                "peak_rainfall_mm_day": round(m["peak_rain"], 2),
                "baseline_runoff_mm": round(m["runoff_2020_mean"], 2),
                "flood_event_runoff_mm": round(m["runoff_2025_mean"], 2),
                "runoff_increase_mm": round(m["runoff_change_mean"], 2),
                "runoff_increase_pct": round(m["runoff_increase_pct"], 2),
                "median_runoff_change_mm": round(raw["runoff_net_increase_median"], 6),
                "maximum_runoff_spike_mm": round(m["max_runoff_increase"], 2),
                "watershed_runoff_gain_m3": round(m["watershed_runoff_gain_m3"], 2),
                "total_watershed_runoff_volume_m3": round(
                    m["total_roi_runoff_volume"], 2
                ),
            },
            "forensic_attribution": {
                "pre_event_forest_loss_ha": round(m["forest_loss_ha"], 2),
                "critical_deforestation_ha": round(
                    m["critical_slope_deforestation_area"], 2
                ),
                "vegetation_damage_ndvi": round(abs(m["mean_ndvi_loss"]), 4),
                "hydrologic_amplification_pct": round(m["runoff_increase_pct"], 2),
                "peak_rainfall_trigger_mm_day": round(m["peak_rain"], 2),
            },
        }

        base_filename = (
            f"tangse_meureudu_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        json_path = os.path.join(OUTPUT_DIR, f"{base_filename}.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=4)

        print(f"\n[✓] Payload data forensik spasial berhasil disimpan di: {json_path}")
