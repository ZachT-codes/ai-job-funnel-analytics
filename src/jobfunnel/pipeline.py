"""End-to-end pipeline: raw CSV -> clean -> SQLite -> features -> model -> report."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow `python -m jobfunnel.pipeline` from the project root without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jobfunnel import clean, config, features, generate, ingest, metrics, model, report, store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("jobfunnel")


def run(generate_data: bool, rows: int) -> Path:
    settings = config.get_settings()
    config.ensure_dirs(settings)

    if generate_data or not settings.raw_csv.exists():
        settings = config.Settings(
            **{**settings.__dict__, "sample_rows": rows}
        )
        generate.write_sample_csv(settings)

    raw = ingest.load_applications(settings.raw_csv)
    cleaned = clean.clean_applications(raw)
    store.load_to_sqlite(cleaned, settings.db_path)
    df = store.read_from_sqlite(settings.db_path)

    funnel = metrics.funnel_metrics(df)
    X, y, feature_names = features.build_features(df, target=settings.target)
    log.info("Feature matrix: %s rows x %s cols", *X.shape)

    model_results = model.train_response_model(
        X, y, test_size=settings.test_size, random_state=settings.random_state
    )
    log.info("Model metrics: %s", model_results["metrics"])

    chart_source = settings.outputs_dir / "response_by_source.png"
    chart_pay = settings.outputs_dir / "pay_vs_response.png"
    report.plot_response_by_source(df, chart_source)
    report.plot_pay_vs_response(df, chart_pay)
    report_path = report.write_markdown_report(
        funnel, model_results, [chart_source, chart_pay], settings.report_path
    )

    print("\nDone.")
    print(f"  Applications: {funnel['total_applications']}")
    print(f"  Response rate: {funnel['overall_response_rate']:.1%}")
    print(f"  ROC-AUC: {model_results['metrics']['roc_auc']}")
    print(f"  Report: {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the job-funnel analytics pipeline.")
    parser.add_argument("--generate", action="store_true", help="Regenerate synthetic data.")
    parser.add_argument("--rows", type=int, default=400, help="Synthetic rows to generate.")
    args = parser.parse_args()
    run(generate_data=args.generate, rows=args.rows)


if __name__ == "__main__":
    main()
