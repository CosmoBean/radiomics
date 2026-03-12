# MU-Glioma-Post QC Notes

## Initial Audit Snapshot

- Audit date: 2026-03-10
- Patients discovered: 203
- Timepoints discovered: 596
- NIfTI files discovered: 2978
- Exact modalities present: `t1n`, `t1c`, `t2f`, `t2w`, `tumor_mask`
- The 5th file is `tumorMask`, which is a multi-class segmentation mask rather than a binary mask.
- Complete 5-file timepoints: 594
- 4-file timepoints missing only `tumor_mask`: `PatientID_0187/Timepoint_3`, `PatientID_0191/Timepoint_1`
- Read failures: 0

## Geometry Findings

- All readable images and masks have shape `240 x 240 x 155`.
- All readable images and masks have spacing `1.0 x 1.0 x 1.0`.
- All readable images and masks are oriented `LPS`.
- All readable images and masks share the same affine hash in the manifest.
- No shape, spacing, orientation, or affine mismatches were detected for the 594 timepoints with masks.

## Mask Findings

- Observed mask label sets are multi-class and include `0|1|2|3|4` and subsets such as `0|2|3|4`, `0|1|2|3`, `0|2|4`, and `0|2`.
- No empty masks were detected.
- Because masks are multi-class, downstream preprocessing needs an explicit policy for:
  - preserving labels for region-aware work
  - collapsing labels to a binary whole-tumor mask for simpler baselines

## Intensity Findings

- Intensities are not globally normalized or z-scored.
- Background is heavily zero-padded, with mean nonzero fraction around `0.157` across modalities.
- Dynamic ranges differ materially by modality, which supports per-modality normalization during preprocessing.

## Random QC Sample

- Sample count: 12 complete timepoints
- Sampling seed: `20260310`
- Sample list: [sampled_timepoints.csv](/project/community/sbandred/mu-glioma/qc/sampled_timepoints.csv)
- Panels: [random_case_panels](/project/community/sbandred/mu-glioma/qc/random_case_panels)
- Overlays: [mask_overlays](/project/community/sbandred/mu-glioma/qc/mask_overlays)
- Histograms: [histograms](/project/community/sbandred/mu-glioma/qc/histograms)

## Spot Checks

- Visual spot check of [PatientID_0043_Timepoint_1_overlays.png](/project/community/sbandred/mu-glioma/qc/mask_overlays/PatientID_0043_Timepoint_1_overlays.png) shows the multi-class tumor mask aligned with lesion anatomy on both `t1c` and `t2w`.
- Visual spot check of [PatientID_0152_Timepoint_2_overlays.png](/project/community/sbandred/mu-glioma/qc/mask_overlays/PatientID_0152_Timepoint_2_overlays.png) shows the tumor bounding box and overlay localized to the expected inferior-right lesion region.
- Visual spot check of [PatientID_0043_Timepoint_1_panel.png](/project/community/sbandred/mu-glioma/qc/random_case_panels/PatientID_0043_Timepoint_1_panel.png) confirms that the four anatomical modalities have consistent field of view and gross anatomical alignment.
- Visual spot check of [PatientID_0043_Timepoint_1_hist.png](/project/community/sbandred/mu-glioma/qc/histograms/PatientID_0043_Timepoint_1_hist.png) confirms modality-specific nonzero intensity distributions rather than standardized intensities.

## Implications For Preprocessing

- Reorientation and resampling are not currently justified.
- The base pipeline can stay in native geometry and focus on:
  - modality completeness checks
  - mask-preserving load logic
  - per-modality nonzero-voxel normalization
  - optional ROI crop generation
- Before model training, decide whether tasks will use:
  - original multi-class tumor labels
  - a derived binary whole-tumor mask
