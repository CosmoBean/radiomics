# SHAP And Feature Presentation Notes

## Presentation Headline

The best observed corrected forward model in this repository is a calibrated logistic regression model using:

- `T1c + FLAIR` radiomics
- `hybrid_basic` clinical inputs
- probability output through `progression_risk_probability`

Peak held-out split:

- ROC AUC: `0.8043`
- balanced accuracy: `0.7243`
- Brier score: `0.1542`
- source run: `results/repeated_forward_hybrid_basic_corrected/seed_62`

## Exactly What Features Went Into The Peak Model

The exported peak model selected `48` features total:

- `23` `T1c` radiomics features
- `19` `FLAIR` radiomics features
- `6` clinical biomarker features

Clinical features present in the selected model:

- `clin_idh1_mutation__2`
- `clin_atrx_mutation__0`
- `clin_idh2_mutation__2`
- `clin_mgmt_methylation__4`
- `clin_atrx_mutation__2`
- `clin_1p_19q__1`

Important note for presentation:

- these clinical variables are dataset-coded one-hot categories
- the suffixes such as `__0`, `__1`, `__2`, `__4` reflect the dataset encoding, not a ranking of biological severity
- for a clinician-facing slide, it is better to describe these as molecular marker status features rather than over-interpret the numeric suffix itself

Representative `T1c` features selected:

- `t1c_log-sigma-2-0-mm-3D_firstorder_Maximum`
- `t1c_glszm_GrayLevelVariance`
- `t1c_wavelet-LLH_glszm_GrayLevelNonUniformityNormalized`
- `t1c_log-sigma-3-0-mm-3D_firstorder_Variance`
- `t1c_wavelet-LLL_glcm_ClusterTendency`
- `t1c_shape_Sphericity`

Representative `FLAIR` features selected:

- `flair_log-sigma-0-5-mm-3D_glszm_LowGrayLevelZoneEmphasis`
- `flair_wavelet-LLL_glszm_SmallAreaHighGrayLevelEmphasis`
- `flair_wavelet-LHH_glcm_SumAverage`
- `flair_glszm_SmallAreaHighGrayLevelEmphasis`
- `flair_wavelet-LLL_glszm_ZoneEntropy`
- `flair_wavelet-HHL_glcm_MaximumProbability`

## Why These Features Matter

At a high level, the selected radiomics features fall into a few interpretable buckets:

- intensity extremes:
  - examples: `firstorder_Maximum`, `firstorder_Minimum`, `firstorder_Variance`
- heterogeneity / non-uniformity:
  - examples: `GrayLevelVariance`, `GrayLevelNonUniformityNormalized`, `ZoneEntropy`
- local structural complexity:
  - examples: `ClusterTendency`, `Busyness`, `Complexity`
- zone / run emphasis:
  - examples: `SmallAreaHighGrayLevelEmphasis`, `LowGrayLevelZoneEmphasis`
- shape:
  - example: `Sphericity`

This means the model is not relying on one simple size measure alone. It is learning from:

- how heterogeneous the enhancing tumor looks on `T1c`
- how structured the abnormal background signal looks on `FLAIR`
- whether the molecular context suggests a more or less aggressive biology

## Baseline SHAP Story

The repository's original SHAP output came from the radiomics-only LightGBM baseline in `results/summary.json`.

Baseline performance:

- ROC AUC: `0.5985`

Top baseline SHAP features:

1. `flair_wavelet-HLL_firstorder_90Percentile`
2. `t1c_log-sigma-3-0-mm-3D_glszm_GrayLevelVariance`
3. `t1c_wavelet-LHH_glszm_SmallAreaHighGrayLevelEmphasis`
4. `t1c_wavelet-LLL_glszm_LowGrayLevelZoneEmphasis`
5. `t2_wavelet-LLL_firstorder_Kurtosis`
6. `t1_log-sigma-3-0-mm-3D_firstorder_Maximum`

Baseline SHAP modality share:

- `T1c`: `36.4%`
- `FLAIR`: `23.3%`
- `T1`: `21.6%`
- `T2`: `18.8%`

Key message:

- even before the hybrid model work, SHAP was already telling us that `T1c` was the strongest imaging modality
- `FLAIR` was the next most useful complementary signal
- this is what motivated narrowing the backbone away from all four modalities toward `T1c + FLAIR`

## Peak Model SHAP Story

For the exported peak model in `model/peak_forward_seed62`, SHAP was computed directly on the held-out split.

Top peak-model SHAP features:

1. `t1c_log-sigma-3-0-mm-3D_firstorder_Variance`
2. `t1c_wavelet-LLL_glcm_ClusterTendency`
3. `t1c_glszm_GrayLevelVariance`
4. `t1c_wavelet-LLL_ngtdm_Complexity`
5. `t1c_wavelet-LLH_glszm_GrayLevelNonUniformityNormalized`
6. `clin_atrx_mutation__0`
7. `t1c_wavelet-HLL_glszm_SmallAreaLowGrayLevelEmphasis`
8. `t1c_log-sigma-1-0-mm-3D_firstorder_Minimum`
9. `flair_glszm_SmallAreaHighGrayLevelEmphasis`
10. `clin_idh2_mutation__2`
11. `t1c_wavelet-HLL_glszm_LowGrayLevelZoneEmphasis`
12. `flair_wavelet-LHH_glcm_SumAverage`

Peak-model SHAP modality share:

- `T1c`: `55.9%`
- `FLAIR`: `31.1%`
- clinical: `13.0%`

Key message:

- the peak model became even more `T1c`-driven than the older baseline
- `FLAIR` still contributes a large second source of signal
- clinical biomarkers matter, but they are not dominating the prediction

## How To Explain SHAP In One Slide

Suggested wording:

- SHAP measures how much each feature moves the model's predicted progression risk away from its average baseline prediction.
- Higher mean absolute SHAP means a feature consistently has a larger effect on predictions.
- In our model, the largest effects came from `T1c` heterogeneity features, followed by `FLAIR` texture features and then a smaller but meaningful molecular biomarker contribution.

## How To Explain The Clinical Features

The most defensible clinical message is not “the model is driven by clinical variables.” The better message is:

- molecular biomarker status provides a stabilizing biologic context
- the strongest stable clinical signals were `ATRX`, `IDH2`, `IDH1`, `1p/19q`, and `PTEN`
- however, the main predictive mass still came from `T1c + FLAIR` radiomics

Stable clinical signals from the forward explainability analysis:

- `clin_atrx_mutation__4`
- `clin_idh2_mutation__2`
- `clin_idh1_mutation__2`
- `clin_1p_19q__2`
- `clin_pten_mutation__1`

Interpretation direction in the coefficient report:

- `ATRX`, `IDH2`, `IDH1`, and `1p/19q` coded indicators were associated with lower predicted progression risk in the fitted forward model
- `PTEN` coded indicator was associated with higher predicted progression risk

## What Happened To The Engineered Report Features

We explicitly engineered:

- ET / NETC / SNFH / RC volume features
- whole-tumor volume
- days from diagnosis to MRI
- days post-radiation therapy
- bidimensional product
- `T1CE / T1` ratio within ET
- RC-adjacent ET fraction
- mean `FLAIR` intensity within SNFH
- WHO grade indicators

These features made the analysis more clinically coherent, but on the controlled fixed-split ablation they did not beat the `hybrid_basic` model:

- `hybrid_basic`: `0.7694`
- `report_core`: `0.7569`
- `report_timing`: `0.7632`
- `report_full`: `0.7493`

Suggested presentation framing:

- “Feature engineering improved interpretability and reporting coherence.”
- “The strongest tested predictive model still came from the lean `T1c + FLAIR + molecular/basic clinical` setup.”

## Probability Output

The final model output should be presented as probability, not just class label.

Use:

- `progression_risk_probability`
- `progression_risk_percent`

Threshold used in the peak split:

- `0.4553`

This makes the system easier to communicate clinically:

- low probability means lower modeled short-term progression risk
- high probability means higher modeled short-term progression risk
- the hard class is secondary and threshold-dependent

## Slide-Ready Takeaways

If you want a concise closing slide, these are the strongest defensible bullets:

- The best observed corrected forward model achieved ROC AUC `0.8043`.
- The model used `T1c + FLAIR` radiomics plus a small molecular clinical block.
- SHAP showed that `T1c` carried most of the predictive signal, with `FLAIR` as the main complementary modality.
- Clinical biomarkers contributed meaningful biologic context but did not dominate the model.
- The engineered report features improved coherence and explainability, even though they did not outperform the lean hybrid model.
- Model output is probability-first through `progression_risk_probability`.
