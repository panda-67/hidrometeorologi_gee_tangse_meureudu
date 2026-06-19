import ee
from datetime import datetime
from src.core.vegetation import VegetationAnalyzer
from config import config


class GajahSatellitePipeline:
    def __init__(self, roi: ee.Geometry):
        self.roi = roi
        self.va = VegetationAnalyzer(roi)

    def execute(self) -> ee.Image:
        # Menghitung rentang waktu nyata dari akhir baseline 2020 ke akhir pre-event 2025
        date_baseline_start = datetime.strptime(config.F_BASELINE_START, "%Y-%m-%d")
        date_post_event_end = datetime.strptime(config.F_POST_EVENT_END, "%Y-%m-%d")

        # Selisih hari dibagi rata-rata jumlah hari dalam setahun (termasuk kabisat)
        delta_days = (date_post_event_end - date_baseline_start).days

        calculated_timeline_years = (
            delta_days / 365.25
        )  # Menghasilkan ~4.90 tahun jika dihitung dari akhir 2020 ke Nov 2025

        col_baseline = self.va.get_collection(
            config.F_BASELINE_START, config.F_BASELINE_END
        )
        col_pre_event = self.va.get_collection(
            config.F_PRE_EVENT_START, config.F_PRE_EVENT_END
        )
        col_post_event = self.va.get_collection(
            config.F_POST_EVENT_START, config.F_POST_EVENT_END
        )

        img_baseline = col_baseline.median().clip(self.roi)
        img_pre_event = col_pre_event.median().clip(self.roi)
        img_post_event = col_post_event.median().clip(self.roi)

        idx_baseline = self.va.calculate_indices(img_baseline)
        idx_pre_event = self.va.calculate_indices(img_pre_event)
        idx_post_event = self.va.calculate_indices(img_post_event)

        bands_baseline = idx_baseline.select(
            ["NDVI", "NDMI"], ["NDVI_baseline", "NDMI_baseline"]
        )
        bands_pre_event = idx_pre_event.select(
            ["NDVI", "NDMI"], ["NDVI_preevent", "NDMI_preevent"]
        )
        bands_post_event = idx_post_event.select(
            ["NDVI", "NDMI"], ["NDVI_postevent", "NDMI_postevent"]
        )

        # Mengukur delta perubahan kesehatan vegetasi hulu
        d_ndvi_degradation = (
            bands_pre_event.select("NDVI_preevent")
            .subtract(bands_baseline.select("NDVI_baseline"))
            .rename("d_NDVI_degradation")
        )
        d_ndvi_destruction = (
            bands_post_event.select("NDVI_postevent")
            .subtract(bands_pre_event.select("NDVI_preevent"))
            .rename("d_NDVI_destruction")
        )

        percentiles = d_ndvi_degradation.reduceRegion(
            reducer=ee.Reducer.percentile([5, 10, 25]),
            geometry=self.roi,
            scale=30,
            maxPixels=1e13,
        )

        p5 = ee.Number(percentiles.get("d_NDVI_degradation_p5"))
        p10 = ee.Number(percentiles.get("d_NDVI_degradation_p10"))
        p25 = ee.Number(percentiles.get("d_NDVI_degradation_p25"))

        ndvi_degradation_mask = d_ndvi_degradation.lt(ee.Number(p10)).rename(
            "NDVI_Degradation_Mask"
        )

        ndvi_degradation_ha_band = (
            ee.Image.pixelArea()
            .multiply(0.0001)
            .updateMask(ndvi_degradation_mask)
            .rename("ndvi_degradation_masif")
        )

        # Satukan semua band ke dalam satu citra
        output_image = ee.Image.cat(
            [
                bands_baseline,
                bands_pre_event,
                bands_post_event,
                d_ndvi_degradation,
                d_ndvi_destruction,
                ndvi_degradation_mask,
                ndvi_degradation_ha_band,
            ]
        )

        return output_image.set(
            {
                "temporal_baseline_years": calculated_timeline_years,
                "ndvi_p5_threshold": p5,
                "ndvi_p10_threshold": p10,
                "ndvi_p25_threshold": p25,
            }
        )
