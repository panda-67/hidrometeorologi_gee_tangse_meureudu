from src.services.forensic_service import ForensicAnalysisService


def main():
    print("====================================================================")
    print("    EVALUATING FORENSIC METRICS ACROSS THE WATERSHED              ")
    print("====================================================================")

    forensic_service = ForensicAnalysisService()

    forensic_service.export_roi()
    pipelines = forensic_service.run_analysis_pipelines()
    raw_stats = forensic_service.execute_geospatial_reduction(pipelines)
    metrics = forensic_service.parse_and_validate_metrics(raw_stats)

    forensic_service.display_report(metrics, raw_stats)
    forensic_service.save_metrics_payload(metrics, raw_stats)


if __name__ == "__main__":
    main()
