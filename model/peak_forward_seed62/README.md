# Peak Forward Model

This folder contains the exported probability-capable bundle for the best observed corrected forward split.

- Source run: `results/repeated_forward_hybrid_basic_corrected/seed_62`
- Modalities: `t1c`, `flair`
- Clinical feature set: `hybrid_basic`
- Output probability column: `progression_risk_probability`

## Files

- `model_bundle.pkl`: pickled classifier, preprocessor, calibrator, and threshold.
- `metadata.json`: exported metrics and selected features.
- `test_predictions.csv`: probability outputs for the held-out split.
- `shap_feature_importance.csv`: test-set mean absolute SHAP values for the exported logistic model.
- `shap_modality_share.csv`: modality-level SHAP aggregation.

## Usage

```python
import pickle
bundle = pickle.load(open('model/peak_forward_seed62/model_bundle.pkl', 'rb'))
```

Use `bundle['classifier']` on preprocessed features for raw probabilities, then pass them through
`bundle['calibrator']` to obtain calibrated progression risk probabilities.
