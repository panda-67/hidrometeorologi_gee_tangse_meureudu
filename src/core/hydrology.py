import ee
from config import config


class HydrologyModeler:
    """
    Menangani analisis medan topografi dan pemodelan hidrologi empiris SCS-CN
    untuk estimasi limpasan permukaan (surface runoff) makro regional.
    """

    def __init__(self, roi: ee.Geometry):
        self.roi = roi
        self.rainfall_mm = config.PEAK_RAINFALL_MM_DAY

        # TR-55 Curve Number - Hydrologic Soil Group C (Kondisi Regional Terverifikasi)
        self.worldcover_cn_map = {
            10: 70,  # Tree Cover
            20: 77,  # Shrubland
            30: 79,  # Grassland
            40: 82,  # Cropland
            50: 92,  # Built-up
            60: 88,  # Bare / Sparse
            70: 75,  # Snow/Ice
            80: 98,  # Water
            90: 85,  # Herbaceous Wetland
            95: 85,  # Mangroves
            100: 85,  # Moss/Lichen
        }

    # =========================================================================
    # TERRAIN COMPONENT (Copernicus DEM GLO-30 & Morfometri)
    # =========================================================================

    def get_dem(self) -> ee.Image:
        """COPERNICUS DEM GLO-30."""
        return (
            ee.ImageCollection("COPERNICUS/DEM/GLO30")
            .select("DEM")
            .mosaic()
            .clip(self.roi)
        )

    def get_elevation(self) -> ee.Image:
        """Alias DEM."""
        return self.get_dem()

    def get_slope(self) -> ee.Image:
        """Slope dalam satuan derajat."""
        dem = self.get_dem()
        return ee.Terrain.slope(dem).rename("Slope")

    def get_aspect(self) -> ee.Image:
        """Aspect lereng."""
        dem = self.get_dem()
        return ee.Terrain.aspect(dem).rename("Aspect")

    def get_hillshade(self) -> ee.Image:
        """Hillshade visualisasi."""
        dem = self.get_dem()
        return ee.Terrain.hillshade(dem).rename("Hillshade")

    # =========================================================================
    # SCS-CN COMPONENT (Curve Number Spasial)
    # =========================================================================

    def worldcover_to_cn(self, landcover: ee.Image) -> ee.Image:
        """ESA WorldCover -> Curve Number (Remap spasial berdasarkan tabel TR-55)."""
        classes = list(self.worldcover_cn_map.keys())
        values = list(self.worldcover_cn_map.values())
        # Default value 75 digunakan jika ada kelas piksel di luar kamus utama
        return landcover.remap(classes, values, 75).rename("CN")

    def apply_degradation_scenario(
        self, cn_image: ee.Image, degradation_mask: ee.Image, degraded_cn: int = 88
    ) -> ee.Image:
        """CN meningkat pada area yang terdegradasi/mengalami forest loss."""
        return cn_image.where(degradation_mask, degraded_cn).rename("CN")

    def potential_retention(self, cn_image: ee.Image) -> ee.Image:
        """S = (25400 / CN) - 254"""
        return ee.Image.constant(25400).divide(cn_image).subtract(254).rename("S")

    def initial_abstraction(self, cn_image: ee.Image) -> ee.Image:
        """Ia = 0.2S"""
        return self.potential_retention(cn_image).multiply(0.2).rename("Ia")

    def get_dynamic_peak_rainfall(self, start_date: str, end_date: str) -> ee.Image:
        """
        Mengekstrak curah hujan puncak (maksimum harian) secara riil
        dari satelit CHIRPS selama rentang waktu bencana.
        """
        chirps = (
            ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterBounds(self.roi)
            .filterDate(start_date, end_date)
            .select("precipitation")
        )

        # Mengambil nilai curah hujan harian tertinggi yang terjadi dalam rentang waktu tersebut
        peak_rainfall_img = chirps.reduce(ee.Reducer.max())

        return peak_rainfall_img.clip(self.roi)

    # =========================================================================
    # RUNOFF COMPONENT (Simulasi Limpasan Permukaan)
    # =========================================================================
    def calculate_runoff(
        self, cn_image: ee.Image, rainfall_image: ee.Image
    ) -> ee.Image:
        """
        SCS-CN Runoff Model dengan Input Curah Hujan Dinamis (ee.Image).
        """
        # BARIS DIUBAH: Tidak lagi menggunakan ee.Image.constant dari config
        s = self.potential_retention(cn_image)
        ia = s.multiply(0.2)

        # Rumus utama TR-55
        numerator = rainfall_image.subtract(ia).pow(2)
        denominator = rainfall_image.add(s.multiply(0.8))
        runoff = numerator.divide(denominator)

        # Kondisi batas hidrologi: Jika P <= Ia, runoff = 0
        final_runoff = runoff.where(rainfall_image.lte(ia), 0)
        return final_runoff.rename("Q_runoff")

    # def calculate_runoff(self, cn_image: ee.Image) -> ee.Image:
    #     """
    #     SCS-CN Runoff Model.
    #     Rumus: Q = (P - Ia)^2 / (P + 0.8S) jika P > Ia, else Q = 0
    #     """
    #     rainfall_image = ee.Image.constant(self.rainfall_mm).clip(self.roi)
    #     s = self.potential_retention(cn_image)
    #     ia = s.multiply(0.2)
    #
    #     # Rumus pembagi TR-55: P - Ia + S = P - 0.2S + S = P + 0.8S
    #     numerator = rainfall_image.subtract(ia).pow(2)
    #     denominator = rainfall_image.add(s.multiply(0.8))
    #     runoff = numerator.divide(denominator)
    #
    #     # Kondisi batas hidrologi: Jika P <= Ia, limpasan air permukaan adalah 0
    #     final_runoff = runoff.where(rainfall_image.lte(ia), 0)
    #     return final_runoff.rename("Runoff_mm")

    def runoff_difference(
        self, runoff_before: ee.Image, runoff_after: ee.Image
    ) -> ee.Image:
        """Kalkulasi delta bersih kenaikan limpasan permukaan."""
        return runoff_after.subtract(runoff_before).rename("Runoff_Change")
