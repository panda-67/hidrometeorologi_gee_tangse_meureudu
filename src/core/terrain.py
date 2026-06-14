import ee


class TerrainAnalyzer:
    """Komputasi parameter fisik topografi wilayah berdasarkan DEMNAS ACEH."""

    def __init__(self, roi: ee.Geometry):
        self.roi = roi
        # Memastikan satu sumber data tunggal resolusi tinggi untuk seluruh analisis
        self.dem = (
            ee.Image("users/nandadata02/DEMNAS-ACEH")
            .select(["b1"], ["elevation"])
            .clip(self.roi)
        )

    def get_dem(self) -> ee.Image:
        """Mengembalikan citra elevasi dasar."""
        return self.dem

    def get_slope(self) -> ee.Image:
        """Slope dalam satuan derajat."""
        return ee.Terrain.slope(self.dem).rename("Slope")

    def get_aspect(self) -> ee.Image:
        """Aspect lereng."""
        return ee.Terrain.aspect(self.dem).rename("Aspect")

    def get_hillshade(self) -> ee.Image:
        """Hillshade visualisasi."""
        return ee.Terrain.hillshade(self.dem).rename("Hillshade")

    def analyze_morfometry(self) -> ee.Image:
        """Menghitung dan menggabungkan peta ketinggian (elevation) dan kemiringan (slope)."""
        return ee.Image.cat([self.dem, self.get_slope()])
