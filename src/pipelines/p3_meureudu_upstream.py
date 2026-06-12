import ee
from src.core.terrain import TerrainAnalyzer
from src.core.landcover import LandCoverAnalyzer
from config import config


class MeureuduUpstreamPipeline:
    def __init__(self, roi: ee.Geometry):
        self.roi = roi
        self.ta = TerrainAnalyzer(roi)
        self.lca = LandCoverAnalyzer(roi)

    def execute(self) -> ee.Image:
        # 1. Hitung Parameter Morfometri Lereng Hulu
        terrain_layers = self.ta.analyze_morfometry()
        slope = terrain_layers.select("slope")

        # 2. Ambil Data Tutupan Lahan Mentah dan Ubah Menjadi Mask Hutan
        lc_2020_raw = self.lca.get_worldcover_2020()
        lc_2025_raw = self.lca.get_dynamic_world(
            config.F_PRE_EVENT_START, config.F_PRE_EVENT_END
        )  # Rentang waktu Pre-Event

        forest_2020 = self.lca.get_forest_mask(lc_2020_raw, source="worldcover")
        forest_2025 = self.lca.get_forest_mask(lc_2025_raw, source="dynamic_world")

        # 3. Identifikasi Piksel Kehilangan Hutan Bersih Bersifat Biner (1 = Kehilangan Hutan)
        # Logic: Ada di rona awal (forest_2020 = 1) DAN tidak ada di masa pra-bencana (forest_2025.Not() = 1)
        loss_preevent = forest_2020.And(forest_2025.Not()).rename(
            "forest_loss_preevent"
        )

        # 4. Kunci Zona Kritis Spasial: Deforestasi Pra-Bencana di Atas Lereng Curam (> 15 Derajat)
        critical_clipping = loss_preevent.updateMask(slope.gt(15)).rename(
            "critical_upstream_deforestation"
        )

        return ee.Image.cat([terrain_layers, loss_preevent, critical_clipping])
