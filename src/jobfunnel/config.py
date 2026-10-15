"""Shared configuration for the job-funnel project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    raw_csv: Path
    db_path: Path
    outputs_dir: Path
    report_path: Path
    random_state: int = 42
    sample_rows: int = 400
    test_size: float = 0.25
    target: str = "got_response"


def get_settings() -> Settings:
    """Build default paths relative to the project root."""
    project_root = Path(__file__).resolve().parents[2]
    outputs_dir = project_root / "outputs"
    return Settings(
        project_root=project_root,
        raw_csv=project_root / "data" / "raw" / "applications.csv",
        db_path=outputs_dir / "jobfunnel.db",
        outputs_dir=outputs_dir,
        report_path=outputs_dir / "report.md",
    )


def ensure_dirs(settings: Settings) -> None:
    settings.raw_csv.parent.mkdir(parents=True, exist_ok=True)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
