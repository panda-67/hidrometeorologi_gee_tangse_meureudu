import ee
from src.core.vegetation import VegetationAnalyzer
from config import config


class GajahSatellitePipeline:
    def __init__(self, roi: ee.Geometry):
        self.roi = roi
        self.va = VegetationAnalyzer(roi)

    def execute(self) -> ee.Image:
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

        # Sesuai logika dokumen lama: Deteksi area degradasi masif (Thresold > 0.1)
        ndvi_degradation_mask = d_ndvi_degradation.lt(-0.1).rename(
            "NDVI_Degradation_Mask"
        )

        return ee.Image.cat(
            [
                bands_baseline,
                bands_pre_event,
                bands_post_event,
                d_ndvi_degradation,
                d_ndvi_destruction,
                ndvi_degradation_mask,
            ]
        )
