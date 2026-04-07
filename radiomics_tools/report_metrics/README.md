# Report Metric Tool Calls

This module exposes small Python functions for the report-style imaging metrics discussed for the postoperative glioma pipeline.

## Label Convention

The utilities assume the MU-Glioma-Post multiclass segmentation labels already used in this repository:

- `1`: enhancing tumor (`ET`)
- `2`: non-enhancing tumor core (`NETC`)
- `3`: surrounding non-enhancing FLAIR hyperintensity (`SNFH`)
- `4`: resection cavity (`RC`)

## Files

- `volumes.py`
  - compartment voxel counting and volume conversion helpers
- `geometry.py`
  - bidimensional tumor burden helpers
- `intensity.py`
  - signal ratio, adjacency, and compartment-intensity helpers
- `io.py`
  - NIfTI loading helpers for tool calls
- `case.py`
  - one-shot wrappers that return all report metrics for a case

## Metric Functions

### Tumor compartment volumes

- `enhancing_tumor_volume_cc(mask_array, spacing)`
  - volume of label `1`
  - units: cubic centimeters
- `non_enhancing_tumor_core_volume_cc(mask_array, spacing)`
  - volume of label `2`
  - units: cubic centimeters
- `snhf_volume_cc(mask_array, spacing)`
  - volume of label `3`
  - units: cubic centimeters
- `resection_cavity_volume_cc(mask_array, spacing)`
  - volume of label `4`
  - units: cubic centimeters
- `whole_tumor_volume_cc(mask_array, spacing)`
  - union volume across labels `1/2/3/4`
  - units: cubic centimeters
- `tumor_compartment_volumes_cc(mask_array, spacing)`
  - returns all five volume metrics in one dictionary

### Size and burden

- `bidimensional_product_cm2(mask_array, spacing, labels=(1,))`
  - maximum in-plane bidimensional product across slices for the selected labels
  - units: square centimeters
- `enhancing_tumor_bidimensional_product_cm2(mask_array, spacing)`
  - ET-specific wrapper used by the current surveillance pipeline

### Spatial and intensity relationships

- `t1ce_to_t1_intensity_ratio_within_et(t1_array, t1ce_array, mask_array)`
  - mean `T1CE` intensity inside `ET` divided by mean `T1` intensity inside `ET`
- `rc_adjacent_et_fraction(mask_array, iterations=1, connectivity=1)`
  - fraction of `ET` voxels touching the one-voxel dilation of `RC`
  - range: `0` to `1`
- `mean_flair_intensity_within_snhf(flair_array, mask_array)`
  - mean `FLAIR` intensity inside `SNFH`
- `mean_intensity_for_labels(image_array, mask_array, labels)`
  - generic compartment mean-intensity helper used by the signal functions

### Case-level wrappers

- `compute_case_report_metrics(mask_array, spacing, t1_array=None, t1ce_array=None, flair_array=None)`
  - returns all requested report metrics from in-memory arrays
- `compute_case_report_metrics_from_paths(CaseMetricPaths(...))`
  - same output, but reads the NIfTI inputs from disk
  - intended entry point for tool calls

## Current Pipeline Mapping

The surveillance script still emits its historical feature names, but now computes them via this package:

- `bidimensional_product_cm2` -> `eng_bd_product_cm2`
- `t1ce_to_t1_intensity_ratio_within_et` -> `eng_t1ce_t1_signal_ratio_within_et`
- `rc_adjacent_et_fraction` -> `eng_rc_adjacent_et_fraction`
- `mean_flair_intensity_within_snhf` -> `eng_mean_flair_signal_within_snhf`

## Example

```python
from pathlib import Path

from radiomics_tools.report_metrics import CaseMetricPaths, compute_case_report_metrics_from_paths

metrics = compute_case_report_metrics_from_paths(
    CaseMetricPaths(
        mask_path=Path("processed/masks/PatientID_0007/Timepoint_2/tumor_mask_multiclass.nii.gz"),
        t1_path=Path("processed/images_normalized/PatientID_0007/Timepoint_2/t1.nii.gz"),
        t1ce_path=Path("processed/images_normalized/PatientID_0007/Timepoint_2/t1c.nii.gz"),
        flair_path=Path("processed/images_normalized/PatientID_0007/Timepoint_2/flair.nii.gz"),
    )
)
```
