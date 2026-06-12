import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Tuple


class GEEConfig(BaseSettings):
    # Meta Project (Nama variabel harus cocok dengan yang ada di kelas atau diizinkan lewat config)
    PROJECT_ID: str = "default-project"
    OUTPUT_DIR: str = "./data/output_metrics"

    # Atribut penampung tambahan agar pydantic mengenali variabel dari .env Anda
    gee_project_id: str = "default-project"
    data_output_dir: str = "./data/output_metrics"
    log_dir: str = "./logs"
    debug_mode: str = "True"

    # TIMELINE FORENSIK MULTI-FASE (Sesuai Cetak Biru Matriks Analisis)
    # 1. Baseline Fase
    F_BASELINE_START: str = "2020-01-01"
    F_BASELINE_END: str = "2020-12-31"

    # 2. Pre-Event Fase (Akumulasi Degradasi Lahan)
    F_PRE_EVENT_START: str = "2025-01-01"
    F_PRE_EVENT_END: str = "2025-11-23"

    # 3. Flood Event Fase (Puncak Hujan & Simulasi Limpasan)
    F_FLOOD_EVENT_START: str = "2025-11-24"
    F_FLOOD_EVENT_END: str = "2025-11-30"

    # 4. Post-Event Fase (Genangan Hilir & Sedimen)
    F_POST_EVENT_START: str = "2025-12-01"
    F_POST_EVENT_END: str = "2026-01-31"

    # Ambang Batas Saintifik (Thresholds)
    CLOUD_PROB_THRESHOLD: int = 20
    NDVI_DEGRADATION_THRESHOLD: float = -0.1

    # Parameter Hidrologi (Skenario Curah Hujan Ekstrem Batas Atas)
    PEAK_RAINFALL_MM_DAY: float = 122.00

    # Jangkar Titik Koordinat Outlet (8 Sistem DAS Hasil Kalibrasi Hulu)
    OUTLET_COORDINATES: List[Tuple[float, float]] = [
        (95.8514831, 5.1869573),  # Lhok Keutapang, Tangse
        (95.9369891, 5.1533933),  # Tiro, Pidie
        (95.9794531, 5.2757963),  # Beureunuen
        (96.0849351, 5.2044313),  # Sarah Panyang
        (96.1381043, 5.2735685),  # Pante Raja
        (96.1813460, 5.2591846),  # Trienggadeng
        (96.2216408, 5.2411777),  # Kuta Trieng
        (96.2535437, 5.2315264),  # Meureudu
    ]

    # Menggunakan SettingsConfigDict bawaan Pydantic v2 untuk melonggarkan pembacaan .env
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",  # <--- MENGIZINKAN INPUT EXTRA DARI .ENV AGAR TIDAK ERROR
        case_sensitive=False,  # <--- Mengabaikan perbedaan huruf besar/kecil antara .env dan python
    )

    # Helper method untuk mengalihkan sinkronisasi parameter dinamis
    def __init__(self, **values):
        super().__init__(**values)
        # Jika pydantic membaca 'gee_project_id' dari .env, timpa nilai PROJECT_ID utama
        if self.gee_project_id and self.gee_project_id != "default-project":
            self.PROJECT_ID = self.gee_project_id
        if self.data_output_dir:
            self.OUTPUT_DIR = self.data_output_dir


config = GEEConfig()
