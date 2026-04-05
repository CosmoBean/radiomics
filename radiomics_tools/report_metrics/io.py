"""Path-based IO helpers for report metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class LoadedCaseData:
    mask_array: np.ndarray
    spacing: tuple[float, float, float]
    t1_array: np.ndarray | None = None
    t1ce_array: np.ndarray | None = None
    flair_array: np.ndarray | None = None


def read_mask(mask_path: str | Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = sitk.ReadImage(str(Path(mask_path)), sitk.sitkUInt8)
    return sitk.GetArrayFromImage(image), image.GetSpacing()


def read_scalar_image(image_path: str | Path) -> np.ndarray:
    image = sitk.ReadImage(str(Path(image_path)), sitk.sitkFloat32)
    return sitk.GetArrayFromImage(image).astype(np.float32, copy=False)


def load_case_data(
    *,
    mask_path: str | Path,
    t1_path: str | Path | None = None,
    t1ce_path: str | Path | None = None,
    flair_path: str | Path | None = None,
) -> LoadedCaseData:
    mask_array, spacing = read_mask(mask_path)
    return LoadedCaseData(
        mask_array=mask_array,
        spacing=spacing,
        t1_array=read_scalar_image(t1_path) if t1_path is not None else None,
        t1ce_array=read_scalar_image(t1ce_path) if t1ce_path is not None else None,
        flair_array=read_scalar_image(flair_path) if flair_path is not None else None,
    )

