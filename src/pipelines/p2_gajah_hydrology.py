import ee
from src.core.landcover import LandCoverAnalyzer
from src.core.hydrology import HydrologyModeler
from config import config


class GajahHydrologyPipeline:
    def __init__(self, roi: ee.Geometry):
        self.roi = roi
        self.lca = LandCoverAnalyzer(roi)
        self.hm = HydrologyModeler(roi)

    def execute(self) -> ee.Image:
        # 1. Ambil data land cover mentah
        lc_2020_raw = self.lca.get_worldcover_2020()
        lc_2025_raw = self.lca.get_dynamic_world(
            config.F_FLOOD_EVENT_START, config.F_FLOOD_EVENT_END
        )

        # 2. Ambil data CURAH HUJAN DINAMIS dari CHIRPS (Rentang fase bencana)
        # Sesuaikan tanggal ini dengan kejadian banjir aktual November 2025
        flood_rainfall_img = self.hm.get_dynamic_peak_rainfall(
            config.F_FLOOD_EVENT_START, config.F_FLOOD_EVENT_END
        )

        # 3. Bangun peta biner forest loss
        forest_2020 = self.lca.get_forest_mask(lc_2020_raw, source="worldcover")
        forest_2025 = self.lca.get_forest_mask(lc_2025_raw, source="dynamic_world")
        loss_preevent = forest_2020.subtract(forest_2025).eq(1)

        # 4. Bangun Matriks Curve Number (CN)
        cn_baseline = self.hm.worldcover_to_cn(lc_2020_raw)
        cn_preevent = self.hm.apply_degradation_scenario(
            cn_image=cn_baseline, degradation_mask=loss_preevent, degraded_cn=88
        )

        # 5. FASE SIMULASI RUNOFF: Masukkan variabel flood_rainfall_img ke dalam argumen
        runoff_baseline = self.hm.calculate_runoff(
            cn_baseline, flood_rainfall_img
        ).rename("Q_simulated_baseline")
        runoff_flood_event = self.hm.calculate_runoff(
            cn_preevent, flood_rainfall_img
        ).rename("Q_actual_floodevent")

        # 6. Hitung Perubahan Limpasan Bersih
        delta_runoff = self.hm.runoff_difference(runoff_baseline, runoff_flood_event)

        # Serta ikut sertakan layer curah hujan dinamis ke dalam hasil untuk dibaca main.py
        return ee.Image.cat(
            [
                runoff_baseline,
                runoff_flood_event,
                delta_runoff.rename("runoff_net_increase"),
                flood_rainfall_img.rename("dynamic_rainfall_peak"),  # Tambah band baru
            ]
        )

    # def execute(self) -> ee.Image:
    #     # 1. Ambil data land cover mentah (Bukan langsung mask biner)
    #     lc_2020_raw = self.lca.get_worldcover_2020()
    #     lc_2025_raw = self.lca.get_dynamic_world("2025-01-01", "2025-11-23")
    #
    #     # 2. Bangun peta biner forest loss dari domain landcover untuk skenario degradasi
    #     forest_2020 = self.lca.get_forest_mask(lc_2020_raw, source="worldcover")
    #     forest_2025 = self.lca.get_forest_mask(lc_2025_raw, source="dynamic_world")
    #     loss_preevent = forest_2020.And(forest_2025.Not())
    #
    #     # 3. Bangun Matriks Curve Number (CN) sesuai Blueprint TR-55
    #     # Baseline menggunakan konversi tabel murni dari ESA WorldCover
    #     cn_baseline = self.hm.worldcover_to_cn(lc_2020_raw)
    #
    #     # Pre-event menimpa nilai CN baseline dengan angka 88 pada area yang mengalami forest loss
    #     cn_preevent = self.hm.apply_degradation_scenario(
    #         cn_image=cn_baseline, degradation_mask=loss_preevent, degraded_cn=88
    #     )
    #
    #     # 4. FASE FLOOD EVENT: Simulasi Runoff Menggunakan Model Hidrologi Baru
    #     runoff_baseline = self.hm.calculate_runoff(cn_baseline).rename(
    #         "Q_simulated_baseline"
    #     )
    #     runoff_flood_event = self.hm.calculate_runoff(cn_preevent).rename(
    #         "Q_actual_floodevent"
    #     )
    #
    #     # 5. Hitung Lonjakan Perubahan Limpasan Bersih dengan fungsi penolak bug
    #     delta_runoff = (
    #         self.hm.cutoff_difference(runoff_baseline, runoff_flood_event)
    #         if "cutoff_difference" in dir(self.hm)
    #         else self.hm.runoff_difference(runoff_baseline, runoff_flood_event)
    #     )
    #
    #     return ee.Image.cat(
    #         [
    #             runoff_baseline,
    #             runoff_flood_event,
    #             delta_runoff.rename("runoff_net_increase"),
    #         ]
    #     )
