# Report Metric Tool Calls

This module exposes small Python functions for the report-style imaging metrics discussed for the postoperative glioma pipeline.

## Label Convention

The utilities assume the MU-Glioma-Post multiclass segmentation labels already used in this repository:

- `1`: enhancing tumor (`ET`)
- `2`: non-enhancing tumor core (`NETC`)
- `3`: surrounding non-enhancing FLAIR hyperintensity (`SNFH`)
- `4`: resection cavity (`RC`)

## Metric Files

- `enhancing_tumor_volume.py`
  - ET volume
- `non_enhancing_tumor_core_volume.py`
  - NETC volume
- `snhf_volume.py`
  - SNFH volume
- `resection_cavity_volume.py`
  - RC volume
- `whole_tumor_volume.py`
  - whole-tumor volume
- `bidimensional_product.py`
  - ET bidimensional product
- `t1ce_to_t1_intensity_ratio_within_et.py`
  - ET-restricted T1CE:T1 signal ratio
- `rc_adjacent_et_fraction.py`
  - RC-adjacent ET fraction
- `mean_flair_intensity_within_snhf.py`
  - SNFH-restricted mean FLAIR intensity

## Support Files

- `volumes.py`
  - shared compartment volume helpers
- `geometry.py`
  - shared bidimensional geometry helpers
- `intensity.py`
  - shared intensity and adjacency helpers
- `io.py`
  - NIfTI loading helpers for tool calls
- `case.py`
  - one-shot wrappers that return all report metrics for a case

## Quick Calculation Summaries

### Enhancing tumor (ET) volume

Calculated by counting all voxels labeled `1` in the multiclass segmentation mask.  
That voxel count is multiplied by voxel volume from image spacing and converted to `cm^3`.

### Non-enhancing tumor core (NETC) volume

Calculated by counting all voxels labeled `2` in the multiclass segmentation mask.  
The count is converted to physical volume using voxel spacing and reported in `cm^3`.

### SNFH volume

Calculated by counting all voxels labeled `3`, corresponding to surrounding non-enhancing FLAIR hyperintensity.  
That count is converted from voxel space to `cm^3` using the image spacing.

### RC volume

Calculated by counting all voxels labeled `4`, corresponding to the resection cavity.  
The voxel count is converted to `cm^3` using the voxel dimensions from the mask image.

### Whole tumor volume

Calculated as the union of all tumor-related labels `1/2/3/4` in the multiclass mask.  
All positive tumor-compartment voxels are summed and converted to `cm^3`.

### Bidimensional product (2D tumor size)

Calculated on the enhancing tumor mask by scanning slices and finding the largest in-plane extent.  
For each slice, height × width is measured in physical units, and the maximum product is reported in `cm^2`.

### T1CE-to-T1 intensity ratio (within ET)

Calculated by taking the mean T1CE intensity inside the ET region and dividing it by the mean T1 intensity inside the same ET region.  
This gives a relative enhancement measure restricted to the enhancing tumor compartment.

### RC-adjacent ET fraction

Calculated by dilating the RC mask by one voxel and checking how much ET overlaps that neighborhood.  
The result is the fraction of ET voxels that directly border the resection cavity, on a `0` to `1` scale.

### Mean FLAIR intensity (within SNFH)

Calculated by taking all FLAIR voxels inside the SNFH compartment and computing their mean intensity.  
This summarizes the average FLAIR signal within the peri-tumoral hyperintense region.

## Metric Functions

### Tumor compartment volumes

- `enhancing_tumor_volume_cc(mask_array, spacing)`
  - file: `enhancing_tumor_volume.py`
  - definition: volume of label `1`
  - units: cubic centimeters
- `non_enhancing_tumor_core_volume_cc(mask_array, spacing)`
  - file: `non_enhancing_tumor_core_volume.py`
  - definition: volume of label `2`
  - units: cubic centimeters
- `snhf_volume_cc(mask_array, spacing)`
  - file: `snhf_volume.py`
  - definition: volume of label `3`
  - units: cubic centimeters
- `resection_cavity_volume_cc(mask_array, spacing)`
  - file: `resection_cavity_volume.py`
  - definition: volume of label `4`
  - units: cubic centimeters
- `whole_tumor_volume_cc(mask_array, spacing)`
  - file: `whole_tumor_volume.py`
  - definition: union volume across labels `1/2/3/4`
  - units: cubic centimeters
- `tumor_compartment_volumes_cc(mask_array, spacing)`
  - returns all five volume metrics in one dictionary

### Size and burden

- `enhancing_tumor_bidimensional_product_cm2(mask_array, spacing)`
  - file: `bidimensional_product.py`
  - definition: maximum in-plane ET bidimensional product across slices
  - units: square centimeters

### Spatial and intensity relationships

- `t1ce_to_t1_intensity_ratio_within_et(t1_array, t1ce_array, mask_array)`
  - file: `t1ce_to_t1_intensity_ratio_within_et.py`
  - definition: mean `T1CE` intensity inside `ET` divided by mean `T1` intensity inside `ET`
- `rc_adjacent_et_fraction(mask_array, iterations=1, connectivity=1)`
  - file: `rc_adjacent_et_fraction.py`
  - fraction of `ET` voxels touching the one-voxel dilation of `RC`
  - range: `0` to `1`
- `mean_flair_intensity_within_snhf(flair_array, mask_array)`
  - file: `mean_flair_intensity_within_snhf.py`
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
