"""Charts and a Markdown report: the part a hiring manager actually reads."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_response_by_source(df: pd.DataFrame, out_path: Path) -> Path:
    rates = df.groupby("source")["got_response"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rates.plot(kind="barh", ax=ax, color="#2563eb")
    ax.set_xlabel("Response rate")
    ax.set_title("Response rate by application source")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_pay_vs_response(df: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, group in df.groupby("got_response"):
        ax.hist(
            group["pay_usd_hr"],
            bins=20,
            alpha=0.6,
            label="Got response" if label == 1 else "No response",
        )
    ax.set_xlabel("Pay (USD/hr)")
    ax.set_ylabel("Applications")
    ax.set_title("Pay distribution: responded vs. ignored")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def write_markdown_report(
    metrics: dict,
    model_results: dict,
    chart_paths: list[Path],
    out_path: Path,
) -> Path:
    m = model_results["metrics"]
    top_features = model_results["feature_importance"].head(8)

    lines = [
        "# Job-Funnel Analytics Report",
        "",
        f"- Applications tracked: **{metrics['total_applications']}**",
        f"- Overall response rate: **{metrics['overall_response_rate']:.1%}**",
        f"- Median days to response: **{metrics['median_days_to_response']}**",
        "",
        "## Response rate by source",
        "",
        metrics["response_by_source"].to_markdown(),
        "",
        "## Response rate by role type",
        "",
        metrics["response_by_role"].to_markdown(),
        "",
        "## Response-prediction model (logistic regression)",
        "",
        f"- Accuracy: **{m['accuracy']}** · Precision: **{m['precision']}** "
        f"· Recall: **{m['recall']}** · F1: **{m['f1']}** · ROC-AUC: **{m['roc_auc']}**",
        f"- Trained on {m['n_train']} applications, evaluated on {m['n_test']} held-out.",
        "",
        "### Strongest predictors",
        "",
        top_features.to_markdown(index=False),
        "",
        "## Charts",
        "",
    ]
    for chart in chart_paths:
        lines.append(f"![{chart.stem}]({chart.name})")
    lines.append("")
    lines.append(
        "_Synthetic demo data. Swap in a real export and the same pipeline runs unchanged._"
    )

    out_path.write_text("\n".join(lines))
    return out_path
