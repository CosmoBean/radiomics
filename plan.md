oma Preprocessing Plan

## Goal

Build a clean, reproducible preprocessing pipeline for the MU-Glioma-Post dataset so the data can be used reliably for:

- radiomics + small classical ML models
- 2D deep learning baselines
- 3D deep learning baselines
- future longitudinal modeling

The dataset already appears to be organized into:

- `PatientID_xxxx/Timepoint_y`
- `.nii.gz` files
- likely modality files such as `brain_t1`, `brain_t1c`, `brain_t2`
- tumor masks for most timepoints

This means the job is **not raw medical-image conversion from scratch**.  
The job is to **audit, validate, standardize, and prepare experiment-ready outputs**.

---

## Core principles

1. **Do not split by slice**
2. **Do not split by timepoint**
3. **Always split by patient**
4. **Do not assume the dataset is perfectly processed just because it is in NIfTI**
5. **Do not apply heavy preprocessing unless audit results justify it**
6. **Every transformation must be reproducible and logged**
7. **Create a manifest first, then preprocess**

---

## Current assumptions from inspection

From the initial dataset inspection:

- data is already in `.nii.gz` format
- data is organized by patient and timepoint
- most timepoints appear to contain 5 files
- some timepoints contain 4 files
- filenames suggest brain-only or brain-cropped images
- tumor masks seem available for almost all timepoints
- modalities likely include T1, T1c, T2, and possibly one more modality or derived image

These assumptions must be verified before preprocessing decisions are frozen.

---

## Phase 1: Dataset audit

### Objective

Build a dataset manifest that describes every usable file.

### Required output

Create a CSV manifest with one row per file containing:

- `patient_id`
- `timepoint`
- `file_name`
- `file_path`
- `modality`
- `is_mask`
- `shape`
- `spacing`
- `dtype`
- `orientation`
- `affine_hash` or affine summary
- `min_intensity`
- `max_intensity`
- `mean_intensity`
- `std_intensity`
- `nonzero_fraction`

### Questions the audit must answer

1. What are the exact modalities present?
2. What is the 5th file in most timepoints?
3. Which timepoints are missing files?
4. Are tumor masks present for every timepoint?
5. Are image and mask shapes aligned?
6. Are voxel spacings consistent?
7. Are orientations consistent?
8. Are intensities already normalized or still raw?
9. Are there empty or corrupted masks?
10. Are there obvious duplicate or malformed files?

### Deliverables

- `metadata/manifest.csv`
- `metadata/timepoint_summary.csv`
- `metadata/missingness_report.csv`

---

## Phase 2: Visual quality control

### Objective

Verify that metadata and actual image content agree.

### Required checks

For a random sample of timepoints:

- visualize the center slices of each modality
- overlay tumor mask on T1c
- overlay tumor mask on T2
- inspect intensity histograms per modality
- inspect bounding boxes of tumor masks
- verify that masks are spatially aligned

### Deliverables

- `qc/random_case_panels/`
- `qc/mask_overlays/`
- `qc/histograms/`
- `qc/qc_notes.md`

### Stop conditions

Do not move forward until:

- masks visibly align with anatomy
- modality naming is understood
- missing-file patterns are documented
- no obvious orientation or spacing issues remain unexplained

---

## Phase 3: Freeze preprocessing strategy

Preprocessing should only be finalized after the audit.

### Likely base pipeline

If the audit confirms the dataset is already reasonably processed, use:

1. verify modality file pairing
2. verify image-mask alignment
3. standardize orientation if needed
4. resample only if spacing varies materially
5. normalize intensities per modality
6. optionally crop to foreground / brain box
7. optionally derive tumor ROI crops
8. save processed outputs and manifest

### Likely unnecessary at first

Do not add these unless audit proves they are needed:

- DICOM conversion
- blind skull stripping
- aggressive denoising
- N4 bias correction everywhere
- arbitrary resizing to ImageNet-like shapes
- slice extraction before manifest and QC are complete

---

## Phase 4: Preprocessing details

### 4.1 Modality harmonization

For each patient-timepoint, identify the set of available modalities.

Target canonical naming:

- `t1`
- `t1c`
- `t2`
- `flair` or other verified modality
- `tumor_mask`

If a modality is missing:

- log it in the manifest
- do not silently substitute or discard without a rule

### 4.2 Orientation standardization

Check all images and masks for consistent anatomical orientation.

If inconsistent:

- reorient all images to one canonical convention
- apply the exact same spatial handling to masks

### 4.3 Spacing standardization

Inspect voxel spacing across all volumes.

If spacing is already highly consistent:
- keep native spacing

If spacing varies substantially:
- resample images to a common spacing
- use interpolation appropriate for images
- use nearest-neighbor interpolation for masks only

### 4.4 Intensity normalization

Because MRI intensities are not standardized, normalize per modality.

Recommended first-pass normalization:

- compute foreground or nonzero mask
- clip to robust percentiles if needed
- z-score normalize within nonzero voxels

Store normalization settings clearly.

### 4.5 Cropping

Create two processed variants if useful:

#### Variant A: Full brain / full field of view
Use when:
- testing full-image baselines
- preserving broader context

#### Variant B: Tumor-centered ROI
Use when:
- building radiomics features
- testing small ROI-based classifiers

Do not destroy the original full-volume version.

### 4.6 Mask validation

For each tumor mask:

- verify same shape as paired reference image
- verify affine compatibility
- verify mask is not empty
- verify mask values are sane
- compute tumor voxel count

Flag problematic cases.

---

## Phase 5: Data splits

### Rule

All splits must be **patient-level**.

### Why

The dataset is longitudinal.  
If different timepoints from the same patient appear in different splits, the evaluation is contaminated.

### Split outputs

Create:

- `metadata/train_patients.txt`
- `metadata/val_patients.txt`
- `metadata/test_patients.txt`
- `metadata/splits.csv`

### Recommended first split

- 70% train
- 15% val
- 15% test

If label balance matters, stratify at the patient level if possible.

---

## Phase 6: Processed dataset outputs

Create a clean processed directory structure.

```text
processed/
├── images_native/
├── images_reoriented/
├── images_resampled/
├── images_normalized/
├── roi_tumor/
├── masks/
├── radiomics_inputs/
└── manifests/
