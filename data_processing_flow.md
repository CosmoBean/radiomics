# Data Processing Flow

## Raw Data

The raw dataset contains postoperative MRI studies organized by patient and timepoint. Each timepoint includes:

- `brain_t1n`: T1 pre-contrast
- `brain_t1c`: T1 post-contrast
- `brain_t2f`: FLAIR
- `brain_t2w`: T2
- `tumorMask`: voxelwise lesion segmentation

The package also includes clinical spreadsheets with progression, survival, molecular, and treatment fields.

## What Was Readily Available

The raw package already provided:

- the four MRI modalities
- a segmentation mask
- patient/timepoint organization
- clinical metadata

What it did not provide was a model-ready table of numerical predictors.

## Why Processing Was Needed

The raw data was clinically meaningful, but not directly usable for machine learning. We needed to:

- connect images to the correct clinical labels
- standardize and validate the imaging inputs
- normalize MRI intensities across cases
- convert images into numerical radiomics features
- build a patient-safe train/test split without leakage

## Processing Steps

1. Audit the raw dataset and build manifests.  
Why: turn folders and files into a machine-readable index.

2. Merge imaging records with clinical metadata.  
Why: assign each MRI timepoint the right patient identity and outcome label.

3. Validate modalities, geometry, and masks.  
Why: ensure each case has the expected inputs before feature extraction.

4. Construct the lesion mask from tumor labels `1/2/3` and exclude cavity-only regions.  
Why: restrict analysis to relevant tumor-associated tissue.

5. Apply N4 bias-field correction to each MRI volume.  
Why: reduce smooth intensity inhomogeneity from MRI acquisition.

6. Z-score normalize intensities within the lesion mask.  
Why: make signal values more comparable across scans.

7. Extract PyRadiomics features from each modality.  
Why: convert images into numeric descriptors of shape, intensity, and texture.

8. Concatenate modality-wise features into one row per timepoint.  
Why: build the tabular feature matrix used for modeling.

9. Split data at the patient level, not the scan level.  
Why: prevent leakage across multiple scans from the same patient.

10. Filter, rank, and model the features.  
Why: reduce redundancy and train a predictor of progression risk.

## One-Line Summary

Raw postoperative MRI scans, masks, and clinical spreadsheets were transformed into a cleaned, labeled, patient-aware radiomics table suitable for predictive modeling.
