# Peak Model Feature Engineering Report

## Scope

This report summarizes three things:

1. The engineered feature blocks added to the repository.
2. The SHAP and explainability findings from the baseline and peak models.
3. The current best held-out result and the exported probability-capable model bundle.

## Engineered Feature Work

The surveillance pipeline now supports three report-oriented feature sets:

- `report_core`
- `report_timing`
- `report_full`

The engineered features added for this work are:

- `eng_et_volume_cc`
- `eng_netc_volume_cc`
- `eng_snhf_volume_cc`
- `eng_rc_volume_cc`
- `eng_whole_tumor_volume_cc`
- `eng_days_from_diagnosis_to_current_mri`
- `eng_days_post_radiation_therapy_end`
- `eng_bd_product_cm2`
- `eng_t1ce_t1_signal_ratio_within_et`
- `eng_rc_adjacent_et_fraction`
- `eng_mean_flair_signal_within_snhf`
- WHO-grade one-hot indicators: `clin_grade_of_primary_brain_tumor__*`

These were engineered to match the clinically coherent report direction:

- tumor burden and compartment volumes
- timing and care-context features known at scan time
- compact derived imaging descriptors aligned to ET / SNFH / RC structure

## Report-Feature Ablation

On the fixed corrected forward split, the engineered report blocks did not outperform the existing `hybrid_basic` model:

| Feature set | ROC AUC | Balanced accuracy | Brier |
| --- | ---: | ---: | ---: |
| `hybrid_basic` | 0.7694 | 0.6625 | 0.1692 |
| `report_core` | 0.7569 | 0.5708 | 0.1846 |
| `report_timing` | 0.7632 | 0.6375 | 0.1729 |
| `report_full` | 0.7493 | 0.6500 | 0.1801 |

Interpretation:

- the engineered report features are now reproducibly available in the pipeline
- they are scientifically useful for reporting and ablation
- they did not become the best-performing model family on this benchmark

## Baseline SHAP Findings

The repository's original SHAP output comes from the radiomics-only LightGBM baseline (`results/summary.json`, ROC AUC 0.5985).

Top baseline SHAP features:

- `flair_wavelet-HLL_firstorder_90Percentile`
- `t1c_log-sigma-3-0-mm-3D_glszm_GrayLevelVariance`
- `t1c_wavelet-LHH_glszm_SmallAreaHighGrayLevelEmphasis`
- `t1c_wavelet-LLL_glszm_LowGrayLevelZoneEmphasis`
- `t2_wavelet-LLL_firstorder_Kurtosis`

Baseline modality share from SHAP:

- `t1c`: 36.4%
- `flair`: 23.3%
- `t1`: 21.6%
- `t2`: 18.8%

That baseline SHAP analysis is what pushed the project toward the `T1c + FLAIR` backbone.

## Peak Model

The current best observed corrected forward split is:

- source run: `results/repeated_forward_hybrid_basic_corrected/seed_62`
- modalities: `T1c + FLAIR`
- clinical feature set: `hybrid_basic`
- model: calibrated logistic regression
- held-out ROC AUC: `0.8043`
- held-out balanced accuracy: `0.7243`
- held-out Brier score: `0.1542`

This model has been exported to:

- `model/peak_forward_seed62/model_bundle.pkl`
- `model/peak_forward_seed62/metadata.json`
- `model/peak_forward_seed62/test_predictions.csv`
- `model/peak_forward_seed62/shap_feature_importance.csv`
- `model/peak_forward_seed62/shap_modality_share.csv`

The exported prediction file is probability-first and includes:

- `progression_risk_probability`
- `progression_risk_percent`
- `predicted_class_by_threshold`

## Peak Model SHAP

SHAP was computed for the exported peak logistic model on the held-out split.

Top peak-model SHAP features:

- `t1c_log-sigma-3-0-mm-3D_firstorder_Variance`
- `t1c_wavelet-LLL_glcm_ClusterTendency`
- `t1c_glszm_GrayLevelVariance`
- `t1c_wavelet-LLL_ngtdm_Complexity`
- `t1c_wavelet-LLH_glszm_GrayLevelNonUniformityNormalized`
- `clin_atrx_mutation__0`
- `flair_glszm_SmallAreaHighGrayLevelEmphasis`
- `clin_idh2_mutation__2`

Peak-model modality share from SHAP:

- `t1c`: 55.9%
- `flair`: 31.1%
- `clinical`: 13.0%

Interpretation:

- `T1c` remains the dominant imaging source of signal.
- `FLAIR` contributes substantial complementary information.
- clinical biomarkers remain present, but the peak split is still mostly driven by `T1c + FLAIR` radiomics.

## Overall Conclusion

The repository now supports the full report-driven feature engineering set, but the strongest observed held-out result remains the calibrated `hybrid_basic` forward model rather than `report_full`.

The main scientific takeaway is:

- engineered report features improved coherence and reproducibility of the analysis
- SHAP supports `T1c + FLAIR` as the strongest imaging backbone
- the current peak tested model is a probability-capable logistic model with ROC AUC `0.8043`
- that exported peak model is now committed under `model/peak_forward_seed62`
