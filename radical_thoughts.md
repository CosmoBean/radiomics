# Radical Thoughts

## Current Best Direction

The strongest observed corrected forward result in this repository is still the calibrated `hybrid_basic` logistic model on:

- `T1c + FLAIR`
- curated molecular / basic clinical covariates
- patient-held-out forward surveillance split

Peak held-out result:

- ROC AUC: `0.8043`
- balanced accuracy: `0.7243`
- Brier score: `0.1542`

Source run:

- `results/repeated_forward_hybrid_basic_corrected/seed_62`

## Why This Direction Held Up

The main lesson from the feature-engineering work is that coherence and accuracy were not the same thing.

- We successfully added the full report-driven feature block:
  - region volumes
  - timing features
  - bidimensional product
  - ET intensity ratio
  - RC-adjacent ET fraction
  - mean FLAIR in SNFH
  - WHO grade
- Those features made the analysis more clinically structured.
- But they did not beat the `hybrid_basic` forward model on the corrected benchmark.

So the strongest scientific story is:

- engineered features improved interpretability and reporting coherence
- the best tested model still came from the leaner `hybrid_basic` setup

## SHAP Takeaway

Across the project, SHAP kept pointing us toward the same imaging backbone:

- `T1c` is the dominant signal source
- `FLAIR` is the strongest complementary modality

For the exported peak model, modality-level SHAP share is:

- `T1c`: `55.9%`
- `FLAIR`: `31.1%`
- clinical: `13.0%`

This is the cleanest current interpretation:

- the model is not purely clinical
- it is not purely radiomics-only either
- it is mostly `T1c + FLAIR` radiomics, with molecular context helping at the margins

## Exported Model

The peak tested model is committed under:

- `model/peak_forward_seed62/model_bundle.pkl`

Related files:

- `model/peak_forward_seed62/metadata.json`
- `model/peak_forward_seed62/test_predictions.csv`
- `model/peak_forward_seed62/shap_feature_importance.csv`
- `model/peak_forward_seed62/shap_modality_share.csv`

The prediction output is probability-first:

- `progression_risk_probability`
- `progression_risk_percent`
- `predicted_class_by_threshold`

## Full Report

The fuller writeup is in:

- `reports/peak_model_feature_engineering_report.md`

That report covers:

- all engineered features added
- the report-feature ablation
- baseline SHAP findings
- peak-model SHAP findings
- the exported probability-capable model bundle
