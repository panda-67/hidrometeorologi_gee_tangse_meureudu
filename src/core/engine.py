import os
import ee
import json
from datetime import datetime
from config import config


class GEEEngine:
    """Kelas dasar untuk inisialisasi aman koneksi Google Earth Engine."""

    def __init__(self):
        try:
            ee.Initialize(project=config.PROJECT_ID)
            print(f"[✓] GEE Terhubung Menggunakan Project ID: {config.PROJECT_ID}")
        except Exception as e:
            raise RuntimeError(f"[X] Gagal menginisialisasi GEE: {str(e)}")

    def get_hydro_roi(self) -> ee.Geometry:
        """Delineasi Multi-DAS otomatis berbasis HydroSHEDS Level 12."""
        outlets = ee.Geometry.MultiPoint(
            [list(coord) for coord in config.OUTLET_COORDINATES]
        )
        hydrosheds = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_12")
        multi_watersheds = hydrosheds.filterBounds(outlets)
        return multi_watersheds.union(maxError=1).geometry()

    def export_roi_to_geojson(
        self, roi: ee.Geometry, filename: str = "watershed_roi.geojson"
    ) -> str:
        """
        Mengekspor objek ee.Geometry dari server GEE menjadi berkas GeoJSON lokal
        agar dapat langsung dimuat ke dalam QGIS / ArcGIS.
        """
        print(f"[~] Fetching ROI geometry coordinates from Earth Engine server...")

        # Mengambil informasi spasial geometri dari server GEE ke lokal Python
        roi_info = roi.getInfo()

        # Menentukan direktori penyimpanan
        output_dir = os.path.join("data", "output_metrics")
        os.makedirs(output_dir, exist_ok=True)
        geojson_path = os.path.join(output_dir, filename)

        # Menyusun struktur GeoJSON standar yang dikenali QGIS menggunakan datetime lokal Python
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": roi_info,
                    "properties": {
                        "name": "Watershed ROI",
                        "project": "Geo-Forensic Study",
                        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    },
                }
            ],
        }

        # Menulis data ke dalam berkas .geojson
        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=4)

        print(f"[✓] Geometri ROI berhasil diekspor untuk QGIS di: {geojson_path}")
        return geojson_path

    def safe_extract_metric(self, stats_dict: dict, key: str) -> float or None:
        """
        Mengekstrak nilai GEE reducer secara ketat berdasarkan pencarian kata kunci band.
        Mengembalikan None jika data tidak ditemukan agar tidak memalsukan laporan forensik.
        """
        if not stats_dict:
            return None
        for k, v in stats_dict.items():
            if key in k:
                return v
        return None
