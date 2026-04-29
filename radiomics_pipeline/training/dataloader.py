"""Load the tables used by the training workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class TrainingTables:
    experiment_index: pd.DataFrame
    feature_table: pd.DataFrame | None = None


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return pd.read_csv(resolved)


def load_experiment_index(path: Path) -> pd.DataFrame:
    return _read_csv(path, "Experiment index")


def load_feature_table(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return _read_csv(path, "Feature table")


def load_training_tables(experiment_index_path: Path, feature_table_path: Path | None = None) -> TrainingTables:
    return TrainingTables(
        experiment_index=load_experiment_index(experiment_index_path),
        feature_table=load_feature_table(feature_table_path),
    )
