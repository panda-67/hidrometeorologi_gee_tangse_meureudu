import ee


class TerrainAnalyzer:
    """Komputasi parameter fisik topografi wilayah hulu."""

    def __init__(self, roi: ee.Geometry):
        self.roi = roi
        self.dem = ee.Image("USGS/SRTMGL1_003").clip(self.roi)

    def analyze_morfometry(self) -> ee.Image:
        """Menghitung peta ketinggian dan kemiringan lereng (slope)."""
        slope = ee.Terrain.slope(self.dem)
        return ee.Image.cat([self.dem.rename("elevation"), slope.rename("slope")])
