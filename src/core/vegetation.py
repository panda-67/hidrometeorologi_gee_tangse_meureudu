import ee
from config import config


class VegetationAnalyzer:
    def __init__(self, roi: ee.Geometry):
        self.roi = roi

    def get_collection(self, start_date: str, end_date: str) -> ee.ImageCollection:
        cloud_threshold = config.CLOUD_PROB_THRESHOLD
        s2_sr = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(self.roi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
        )
        s2_clouds = (
            ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
            .filterBounds(self.roi)
            .filterDate(start_date, end_date)
        )
        joined = ee.ImageCollection(
            ee.Join.saveFirst("cloud_mask").apply(
                primary=s2_sr,
                secondary=s2_clouds,
                condition=ee.Filter.equals(
                    leftField="system:index", rightField="system:index"
                ),
            )
        )

        def advanced_mask(img):
            s2_projection = img.select("B4").projection()
            cloud_img = ee.Image(img.get("cloud_mask"))
            cloud_prob = cloud_img.select("probability").setDefaultProjection(
                s2_projection
            )
            is_cloud = cloud_prob.gt(40)
            cloud_shadow_distance = (
                is_cloud.directionalDistanceTransform(180, 50)
                .select(0)
                .setDefaultProjection(s2_projection)
            )
            is_shadow = img.select("B8").lt(1500).And(cloud_shadow_distance.gt(0))
            mask = is_cloud.Or(is_shadow).Not()
            spectral_bands = img.select(["B2", "B3", "B4", "B8", "B11", "B12"])
            scaled_bands = spectral_bands.divide(10000)
            return (
                img.addBands(scaled_bands, overwrite=True)
                .updateMask(mask)
                .copyProperties(img, ["system:time_start"])
            )

        return joined.map(advanced_mask)

    def calculate_indices(self, rgb_nir_img: ee.Image) -> ee.Image:
        ndvi = rgb_nir_img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        ndmi = rgb_nir_img.normalizedDifference(["B8", "B11"]).rename("NDMI")
        return ee.Image.cat([ndvi, ndmi])

    @staticmethod
    def area_hectares(image_mask: ee.Image, roi: ee.Geometry) -> ee.Number:
        """Menghitung luasan area bermasker ke dalam satuan Hektar (ha)."""
        area_img = image_mask.multiply(ee.Image.pixelArea())
        total_area = area_img.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=roi, scale=30, maxPixels=1e13
        )
        # Mengambil nilai pertama hasil reduksi sum secara aman
        return ee.Number(total_area.values().get(0)).divide(10000)
