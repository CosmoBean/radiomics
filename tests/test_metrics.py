from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from radiomics_tools.metrics import (
    CaseMetricPaths,
    compute_case_report_metrics,
    compute_case_report_metrics_from_paths,
    enhancing_tumor_bidimensional_product_cm2,
    rc_adjacent_et_fraction,
    t1ce_to_t1_intensity_ratio_within_et,
    tumor_compartment_volumes_cc,
)


class ReportMetricTests(unittest.TestCase):
    def test_compartment_volumes_cc(self) -> None:
        mask = np.array(
            [
                [[1, 1], [2, 0]],
                [[3, 4], [4, 0]],
            ],
            dtype=np.uint8,
        )
        spacing = (1.0, 1.0, 1.0)

        metrics = tumor_compartment_volumes_cc(mask, spacing)

        self.assertAlmostEqual(metrics["enhancing_tumor_volume_cc"], 0.002)
        self.assertAlmostEqual(metrics["non_enhancing_tumor_core_volume_cc"], 0.001)
        self.assertAlmostEqual(metrics["snhf_volume_cc"], 0.001)
        self.assertAlmostEqual(metrics["resection_cavity_volume_cc"], 0.002)
        self.assertAlmostEqual(metrics["whole_tumor_volume_cc"], 0.006)

    def test_enhancing_tumor_bidimensional_product_cm2(self) -> None:
        mask = np.zeros((2, 5, 5), dtype=np.uint8)
        mask[0, 1:3, 1:4] = 1
        mask[1, 0:4, 0:2] = 1

        result = enhancing_tumor_bidimensional_product_cm2(mask, (1.0, 1.0, 1.0))

        self.assertAlmostEqual(result, 0.08)

    def test_signal_and_adjacency_metrics(self) -> None:
        mask = np.zeros((1, 3, 3), dtype=np.uint8)
        mask[0, 1, 1] = 1
        mask[0, 1, 2] = 1
        mask[0, 1, 0] = 4
        mask[0, 0, 1] = 3

        t1 = np.array([[[0.0, 10.0, 30.0], [0.0, 10.0, 30.0], [0.0, 0.0, 0.0]]], dtype=np.float32)
        t1ce = np.array([[[0.0, 20.0, 60.0], [0.0, 20.0, 60.0], [0.0, 0.0, 0.0]]], dtype=np.float32)
        flair = np.array([[[0.0, 90.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=np.float32)

        ratio = t1ce_to_t1_intensity_ratio_within_et(t1, t1ce, mask)
        adjacency = rc_adjacent_et_fraction(mask)
        metrics = compute_case_report_metrics(
            mask_array=mask,
            spacing=(1.0, 1.0, 1.0),
            t1_array=t1,
            t1ce_array=t1ce,
            flair_array=flair,
        )

        self.assertAlmostEqual(ratio, 2.0)
        self.assertAlmostEqual(adjacency, 0.5)
        self.assertAlmostEqual(metrics["mean_flair_intensity_within_snhf"], 90.0)

    def test_path_wrapper_reads_nifti_inputs(self) -> None:
        mask = np.zeros((1, 2, 2), dtype=np.uint8)
        mask[0, 0, 0] = 1
        mask[0, 0, 1] = 3
        mask[0, 1, 0] = 4

        t1 = np.array([[[5.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
        t1ce = np.array([[[10.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
        flair = np.array([[[0.0, 7.0], [0.0, 0.0]]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            mask_path = tmp / "mask.nii.gz"
            t1_path = tmp / "t1.nii.gz"
            t1ce_path = tmp / "t1c.nii.gz"
            flair_path = tmp / "flair.nii.gz"

            self._write_image(mask, mask_path, sitk.sitkUInt8)
            self._write_image(t1, t1_path, sitk.sitkFloat32)
            self._write_image(t1ce, t1ce_path, sitk.sitkFloat32)
            self._write_image(flair, flair_path, sitk.sitkFloat32)

            metrics = compute_case_report_metrics_from_paths(
                CaseMetricPaths(
                    mask_path=mask_path,
                    t1_path=t1_path,
                    t1ce_path=t1ce_path,
                    flair_path=flair_path,
                )
            )

        self.assertAlmostEqual(metrics["enhancing_tumor_volume_cc"], 0.001)
        self.assertAlmostEqual(metrics["snhf_volume_cc"], 0.001)
        self.assertAlmostEqual(metrics["resection_cavity_volume_cc"], 0.001)
        self.assertTrue(math.isfinite(metrics["bidimensional_product_cm2"]))
        self.assertAlmostEqual(metrics["t1ce_to_t1_intensity_ratio_within_et"], 2.0)
        self.assertAlmostEqual(metrics["mean_flair_intensity_within_snhf"], 7.0)

    @staticmethod
    def _write_image(array: np.ndarray, path: Path, pixel_type: int) -> None:
        image = sitk.GetImageFromArray(array)
        image = sitk.Cast(image, pixel_type)
        image.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(image, str(path))


if __name__ == "__main__":
    unittest.main()
