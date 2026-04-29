# Explainable Radiomics For Postoperative Glioma Surveillance

This repository implements a postoperative glioma surveillance pipeline on MU-Glioma-Post and compares a paper-style radiomics baseline against improved hybrid models.

## Motivation

The goals were:

- reproduce the published postoperative surveillance workflow
- keep the model interpretable
- improve held-out ROC AUC with a cleaner prediction target and better features

## Process

The main entrypoint is [main.py](main.py), and the model-training loader lives in [radiomics_pipeline/training/dataloader.py](radiomics_pipeline/training/dataloader.py).

1. audited and indexed the longitudinal postoperative MRI dataset
2. built lesion masks from labels `1/2/3` and excluded resection-cavity-only regions
3. applied N4 bias correction and lesion-mask z-score normalization
4. extracted PyRadiomics features from `T1`, `T1c`, `T2`, and `FLAIR`
5. evaluated models with patient-held-out splits
6. narrowed the imaging backbone to `T1c + FLAIR`
7. moved from a progression-state framing to a forward-prediction framing
8. added curated molecular and basic clinical covariates to form a hybrid explainable-radiomics model

## Comparative Results

| Setting | Model | Inputs | Held-out design | ROC AUC |
| --- | --- | --- | --- | ---: |
| Christodoulou et al. (baseline paper) | LightGBM-256 | Radiomics, postoperative surveillance | 30 patients / 96 scans | 0.80 |
| Naive initial replication | LightGBM-64 | Radiomics only, paper-style replication | 30 patients / 96 scans | 0.599 |
| Paper-style hybrid attempt | LogReg-48 | `T1c + FLAIR` radiomics + molecular/basic clinical features | 30 patients / 96 scans | 0.621 |
| Forward radiomics-only | LogReg-32 | `T1c + FLAIR` radiomics | 30 patients / 84 scans | 0.674 |
| Earliest-scan hybrid screen | LogReg-48 | `T1c + FLAIR` radiomics + molecular/basic clinical features | 30 patients / 30 scans | **0.873** |
| Calibrated forward hybrid | LogReg-32 | `T1c + FLAIR` radiomics + molecular/basic clinical features | 30 patients / 84 scans | **0.804** |

## Interpretation

- Radiomics-only replication did not recover the target performance.
- Hybrid gains were strongest in the calibrated forward logistic-regression run and the earliest-scan screen, but not on the looser `post_progression` paper-style split.
- `T1c + FLAIR` was the strongest imaging backbone.
- Logistic regression was the most stable tabular model in held-out evaluation.
- The best gains came from adding age, sex, and curated molecular features to the radiomics table.

## Citations

- Christodoulou RC, Vamvouras G, Pitsillos R, Solomou EE, Georgiou MF. *Explainable radiomics with probability calibration for postoperative glioblastoma surveillance*. European Journal of Radiology Artificial Intelligence. 2026;5:100074. doi: [10.1016/j.ejrai.2026.100074](https://doi.org/10.1016/j.ejrai.2026.100074)
- Mahmoud E, Gass J, Dhemesh Y, et al. *MU-Glioma Post: A comprehensive dataset of automated MR multi-sequence segmentation and clinical features*. Scientific Data. 2025;12:1847. doi: [10.1038/s41597-025-06011-7](https://doi.org/10.1038/s41597-025-06011-7)
- Yaseen D, Garrett F, Gass J, et al. *University of Missouri Post-operative Glioma Dataset (MU-Glioma-Post) (Version 1)*. The Cancer Imaging Archive. 2025. doi: [10.7937/7K9K-3C83](https://doi.org/10.7937/7K9K-3C83)

## Layout

- `scripts/prep-data.sh` builds the manifests and processed inputs.
- `scripts/run.sh` runs the calibrated forward hybrid training flow in one go.
- `main.py` is the top-level CLI entrypoint.
- `radiomics_tools/metrics/` contains the reusable engineered metric helpers.
- `radiomics_pipeline/training/` contains the model-side data loading helpers.
- `radiomics_pipeline/workflows/` contains the Python workflow code behind the shell wrappers.
- `models/calibrated/` contains the exported calibrated bundle.
- `configs/` contains the PyRadiomics configuration.
- `tests/` contains the metric unit tests.

This repository contains code and the retained model artifacts, not the raw TCIA data. The recommended direction is the calibrated forward hybrid logistic-regression model: `T1c + FLAIR` radiomics plus curated molecular/basic clinical features.
