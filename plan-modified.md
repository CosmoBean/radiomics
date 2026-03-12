eproduction Plan: Explainable Radiomics with Probability Calibration for Postoperative Glioblastoma Surveillance

**Paper:** Christodoulou et al., *European Journal of Radiology Artificial Intelligence* 5 (2026) 100074  
**Reference Repo:** `https://github.com/georgeDV2002/Explainable-Radiomics-with-Probability-Calibration-for-Postoperative-Glioblastoma-Surveillance`  
**Target AUC:** 0.80 (LightGBM-256, patient-held-out test set)

---

## Phase 0 — Environment & Dependencies

### 0.1 Python Environment
```bash
conda create -n gbm-radiomics python=3.10 -y
conda activate gbm-radiomics
```

### 0.2 Core Packages
```bash
pip install \
  pyradiomics==3.1.0 \
  SimpleITK \
  nibabel \
  numpy pandas scipy scikit-learn \
  lightgbm optuna \
  shap matplotlib seaborn
```

### 0.3 Clone the Authors' Reference Code
```bash
git clone https://github.com/georgeDV2002/Explainable-Radiomics-with-Probability-Calibration-for-Postoperative-Glioblastoma-Surveillance.git
cd Explainable-Radiomics-with-Probability-Calibration-for-Postoperative-Glioblastoma-Surveillance
```
Cross-reference every step below against the repo. If the authors provide config files (e.g., PyRadiomics YAML), use them directly.

### 0.4 Verify Dataset Structure
The MU-Glioma-Post dataset (TCIA) should contain, per patient per timepoint:
- `T1.nii.gz` — T1 pre-contrast
- `T1CE.nii.gz` — T1 post-contrast (gadolinium)
- `T2.nii.gz` — T2-weighted
- `FLAIR.nii.gz` — FLAIR
- `seg.nii.gz` — nnU-Net segmentation mask (labels: ET, NETC, SNFH, RC)
- Clinical metadata CSV with: patient ID, progression status, days to first progression, age, sex, IDH1 mutation, MGMT methylation, treatment info

Confirm the total is **494 timepoint-level samples** across patients, with **380 progression** and **114 no-progression** (Table 1).

---

## Phase 1 — Preprocessing

The TCIA curators already performed resampling, co-registration, and skull-stripping. Verify this, then apply the remaining steps the authors describe.

### 1.1 Verify Pre-existing Preprocessing
For each volume, confirm:
- Isotropic 1 mm³ voxels (check header with `nibabel`)
- Co-registered to SRI24 atlas
- Skull-stripped (no extracranial tissue)
- Segmentation labels present: ET (Enhancing Tissue), NETC (Non-Enhancing Tumor Core), SNFH (Surrounding Non-enhancing FLAIR Hyperintensity), RC (Resection Cavity)

```python
import nibabel as nib
img = nib.load("path/to/T1CE.nii.gz")
print(img.header.get_zooms())  # should be ~(1.0, 1.0, 1.0)
```

### 1.2 N4 Bias Field Correction
Apply N4 correction to every MRI volume (T1, T1CE, T2, FLAIR) using SimpleITK:

```python
import SimpleITK as sitk

def apply_n4(input_path, mask_path, output_path):
    img = sitk.ReadImage(input_path, sitk.sitkFloat32)
    mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(img, mask)
    sitk.WriteImage(corrected, output_path)
```

Use the tumor mask (union of ET + NETC + SNFH) as the mask input.

### 1.3 Z-Score Normalization (Within Lesion Mask)
Critical detail: normalize using **only the voxels inside the tumor mask** (union of ET, NETC, SNFH — NOT including resection cavity).

```python
import numpy as np
import nibabel as nib

def zscore_within_lesion(vol_path, mask_path, output_path):
    vol = nib.load(vol_path)
    data = vol.get_fdata().astype(np.float64)
    mask = nib.load(mask_path).get_fdata()
    
    # Tumor mask = union of ET, NETC, SNFH (labels 1,2,3 — verify exact label IDs)
    tumor_voxels = data[mask > 0]
    mu = tumor_voxels.mean()
    sigma = tumor_voxels.std()
    
    data = (data - mu) / (sigma + 1e-8)
    nib.save(nib.Nifti1Image(data, vol.affine, vol.header), output_path)
```

### 1.4 Construct the Tumor Mask for Extraction
The tumor mask used for radiomic feature extraction is the union of ET + NETC + SNFH (exclude RC). Verify this from the authors' code — the paper says "the tumor mask, which in MU-Glioma-Post represents the union of enhancing tumour, non-enhancing core, and peri-cavitary FLAIR hyperintensity."

```python
# Example: if ET=1, NETC=2, SNFH=3, RC=4
seg = nib.load("seg.nii.gz").get_fdata()
tumor_mask = ((seg == 1) | (seg == 2) | (seg == 3)).astype(np.uint8)
```

---

## Phase 2 — Radiomics Feature Extraction

### 2.1 PyRadiomics Configuration
Create (or use the authors') a PyRadiomics params YAML file with the following settings:

```yaml
imageType:
  Original: {}
  LoG:
    sigma: [0.5, 1.0, 2.0, 3.0]
  Wavelet: {}

featureClass:
  shape:       # 3D ROI descriptors (volume, surface area, sphericity)
  firstorder:  # histogram stats
  glcm:        # co-occurrence textures
  glrlm:       # run-length textures
  glszm:       # size-zone textures
  gldm:        # gray-level dependence textures
  ngtdm:       # neighborhood gray-tone difference textures

setting:
  binWidth: 25                  # verify from repo — standard default
  resampledPixelSpacing: null   # already 1mm isotropic
  interpolator: sitkBSpline
  preCrop: true
  normalize: false              # already z-scored manually
  geometryTolerance: 1e-6
  correctMask: true
  label: 1                      # tumor mask label
```

**Important:** Check the authors' repo for their exact YAML. Grey-level discretization (`binWidth`) and any additional settings (e.g., `force2D`, `force2Ddimension`) must match exactly.

### 2.2 Run Extraction
Extract features for **each modality × each timepoint** using the tumor mask:

```python
import radiomics
from radiomics import featureextractor

extractor = featureextractor.RadiomicsFeatureExtractor("params.yaml")

# For each patient, each timepoint, each modality:
for modality in ["T1", "T1CE", "T2", "FLAIR"]:
    result = extractor.execute(image_path, mask_path)
    # result is an OrderedDict; filter out diagnostic keys
    features = {k: v for k, v in result.items() if not k.startswith("diagnostics_")}
```

### 2.3 Assemble Feature Matrix
- Prefix each feature name with the modality (e.g., `T1CE_wavelet-LLH_glcm_Correlation`)
- Concatenate all 4 modalities per timepoint into a single row
- Expected initial feature count: **~4,892 features** (paper says 4,892 before filtering → 512 after)
- Output: a single DataFrame with shape `(494, ~4892)` plus patient ID, timepoint, and progression label columns

---

## Phase 3 — Train/Test Split (Patient-Level)

### 3.1 Patient-Held-Out Test Set
The paper specifies:
- **Test set:** 96 samples from **30 patients** entirely held out
- **Training/CV set:** 597 samples from the remaining patients
- Total: 693 samples — but Table 1 says 494 timepoints. Reconcile this: the 494 may be usable unique timepoints, or the 597+96=693 may include multiple timepoints per patient. Read the authors' code to clarify exact split logic.

**Critical:** No patient should appear in both train and test. Split by **patient ID**, not by sample.

```python
from sklearn.model_selection import GroupShuffleSplit

# Ensure 30 patients in test
gss = GroupShuffleSplit(n_splits=1, test_size=30_patients_fraction, random_state=SEED)
train_idx, test_idx = next(gss.split(X, y, groups=patient_ids))
```

**Exact patient IDs for the test set:** Check the authors' repo for the specific 30 patients or the random seed used. This is the single biggest source of irreproducibility if not matched.

### 3.2 Verify Class Distribution on Test Set
From the confusion matrix (Fig. 4): test set has **43 no-progression** and **53 progression** samples (96 total). Confirm your split matches this.

---

## Phase 4 — Feature Ranking (on Training Set Only)

### 4.1 Variance Filter
Remove features with variance < 1e-8 (near-constant).

```python
from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=1e-8)
X_train_filtered = vt.fit_transform(X_train)
```

### 4.2 Spearman Correlation Filter
Drop one feature from every pair with |Spearman ρ| ≥ 0.80.

```python
from scipy.stats import spearmanr

corr_matrix, _ = spearmanr(X_train_filtered)
# Iteratively drop features with highest average correlation among pairs > 0.80
```

**Post-filter target:** ~512 features remaining (paper: "512 that remained after filtering the initial 4892").

### 4.3 L1-Logistic Regression + Permutation Importance Ranking
Build a scikit-learn Pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(solver="saga", penalty="l1", max_iter=10000,
                                   C=1.0))  # verify C from repo
])

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

coef_ranks = []
perm_ranks = []

for train_fold, val_fold in skf.split(X_train_512, y_train):
    pipeline.fit(X_train_512[train_fold], y_train[train_fold])
    
    # L1 coefficient magnitudes
    coefs = np.abs(pipeline.named_steps["logreg"].coef_[0])
    
    # Permutation importance on validation fold
    perm = permutation_importance(pipeline, X_train_512[val_fold], y_train[val_fold],
                                  n_repeats=100, random_state=SEED)
    
    coef_ranks.append(coefs)
    perm_ranks.append(perm.importances_mean)
```

### 4.4 Consensus Ranking
For each feature:
1. Compute median absolute L1 coefficient across folds → rank
2. Compute median permutation importance across folds → rank
3. Consensus rank = sum of both ranks
4. Sort ascending (lower = more important)

This produces an ordered feature list. The top 512 are already post-filter; the paper tests subsets of 512, 256, 128, and 64.

---

## Phase 5 — Model Training & Hyperparameter Optimization

### 5.1 Feature Subsets
Create four feature configurations:
- **512 features** (all post-filter)
- **256 features** (top 50% by consensus rank)
- **128 features** (top 25%)
- **64 features** (top 12.5%)

### 5.2 Preprocessing During Optuna Optimization
Within each Optuna trial, apply:
1. Variance threshold (threshold tuned by Optuna)
2. Spearman correlation filter (threshold tuned by Optuna — may be stricter than 0.80)
3. Standard scaling

### 5.3 Patient-Aware 5-Fold Cross-Validation
Use `GroupKFold` or a custom splitter to ensure no patient leaks between folds:

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for fold_train, fold_val in gkf.split(X_train, y_train, groups=patient_ids_train):
    # Train and evaluate
```

### 5.4 LightGBM + Optuna (Primary Model)
```python
import lightgbm as lgb
import optuna

def objective(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "is_unbalance": True,  # or use scale_pos_weight
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "num_leaves": trial.suggest_int("num_leaves", 8, 256),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    
    # Also tune correlation threshold for additional filtering
    corr_threshold = trial.suggest_float("corr_threshold", 0.5, 0.95)
    
    aucs = []
    for fold_train, fold_val in gkf.split(X_train_subset, y_train, groups=patient_ids_train):
        model = lgb.LGBMClassifier(**params)
        model.fit(X_fold_train, y_fold_train)
        y_prob = model.predict_proba(X_fold_val)[:, 1]
        aucs.append(roc_auc_score(y_fold_val, y_prob))
    
    return max(aucs)  # paper says "best over splits"

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=400)
```

**Key detail:** The paper says "The metric used for hyperparameter optimization is the validation fold AUC (best over splits)." This means they report the maximum fold AUC per trial as the objective, not the mean. Verify in repo — this is unusual and may affect results.

### 5.5 Comparison Models
Train with the same CV setup:
- **Logistic Regression:** L2 penalty, class-weighted
- **Random Forest:** class-weighted
- **SVM (RBF kernel):** class-weighted, probability=True

Each should also go through Optuna optimization (400 trials, TPE sampler).

### 5.6 Final Training
Retrain the best LightGBM-256 configuration on the full training set (597 samples) with the best hyperparameters from Optuna.

---

## Phase 6 — Probability Calibration

### 6.1 Collect Out-of-Fold Predictions
During the final 5-fold CV with best hyperparameters, collect validation probabilities for every training sample (each sample appears in exactly one validation fold).

```python
oof_probs = np.zeros(len(X_train))
for fold_train, fold_val in gkf.split(X_train_256, y_train, groups=patient_ids_train):
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train_256[fold_train], y_train[fold_train])
    oof_probs[fold_val] = model.predict_proba(X_train_256[fold_val])[:, 1]
```

### 6.2 Platt Scaling
Fit a logistic regression on (oof_probs → y_train):

```python
from sklearn.calibration import CalibratedClassifierCV
# Or manually:
from sklearn.linear_model import LogisticRegression

calibrator = LogisticRegression()
calibrator.fit(oof_probs.reshape(-1, 1), y_train)
```

### 6.3 Apply Calibration to Test Set
```python
test_probs_raw = final_model.predict_proba(X_test_256)[:, 1]
test_probs_calibrated = calibrator.predict_proba(test_probs_raw.reshape(-1, 1))[:, 1]
```

### 6.4 Threshold Selection
The paper says: "the threshold is selected by minimizing false positives and false negatives." This likely means optimizing the threshold on the calibrated OOF probabilities for balanced accuracy or Youden's J:

```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train, calibrated_oof_probs)
optimal_idx = np.argmax(tpr - fpr)  # Youden's J
optimal_threshold = thresholds[optimal_idx]
```

### 6.5 Evaluate Brier Score
```python
from sklearn.metrics import brier_score_loss

brier_raw = brier_score_loss(y_test, test_probs_raw)
brier_cal = brier_score_loss(y_test, test_probs_calibrated)

# Expected: raw ~0.093, calibrated ~0.088
```

---

## Phase 7 — Evaluation on Held-Out Test Set

### 7.1 Target Metrics (LightGBM-256)
| Metric | Expected Value |
|--------|----------------|
| AUC | 0.80 [0.69–0.89] |
| Class 0 Precision | 0.82 |
| Class 0 Recall | 0.63 |
| Class 0 F1 | 0.71 |
| Class 1 Precision | 0.75 |
| Class 1 Recall | 0.89 |
| Class 1 F1 | 0.81 |
| False Positives | 16 |
| False Negatives | 6 |
| Brier (raw) | 0.093 |
| Brier (calibrated) | 0.088 |

### 7.2 Confusion Matrix Verification
```
                Pred 0    Pred 1
True Class 0:     27        16
True Class 1:      6        47
```

### 7.3 ROC Curve
Plot the ROC curve with 95% CI (bootstrap or DeLong method). The 95% CI should be approximately [0.69, 0.89].

---

## Phase 8 — Decision Curve Analysis

### 8.1 Net Benefit Calculation
On the test set, compute net benefit for threshold probabilities from 0.01 to 0.99:

```python
def net_benefit(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    n = len(y_true)
    return tp / n - fp / n * (threshold / (1 - threshold))
```

### 8.2 Expected Behavior
- Model curve above "treat-all" and "treat-none" from pt ≈ 0.20 to pt ≈ 0.70
- Net benefit gain of 0.05–0.15 relative to treat-none in that range
- Above pt ≈ 0.75, model converges to treat-none

---

## Phase 9 — SHAP Explainability

### 9.1 Compute SHAP Values
```python
import shap

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_all_256)  # or X_test_256
# For LightGBM binary: shap_values may be a single array (log-odds)
```

### 9.2 Verify Top Features Match Table 3
The top contributing features by mean |SHAP| should include:

| Index | Modality | Filter | Family | Feature |
|-------|----------|--------|--------|---------|
| 0 | T2 | LoG σ=1mm | GLSZM | Small Area Emphasis |
| 5 | FLAIR | Original | GLSZM | Small Area Low Gray Level Emphasis |
| 7 | T2 | Wavelet HLL | 1st Order | Median |
| 32 | T1CE | Wavelet LLH | GLCM | Correlation |
| 43 | T2 | Wavelet LHL | GLCM | Inverse Variance |
| 55 | T1CE | Original | GLSZM | Small Area Low Gray Level Emphasis |
| 101 | T2 | Wavelet LHH | GLCM | LMC2 |
| 105 | T1 | Wavelet LHH | GLCM | Inverse Variance |
| 128 | T2 | LoG σ=0.5mm | GLSZM | Small Area Emphasis |
| 149 | FLAIR | Wavelet LHH | 1st Order | Skewness |
| 203 | T1CE | Wavelet HHH | 1st Order | Median |
| 240 | T1CE | Wavelet LHL | GLCM | Correlation |

### 9.3 Modality-Level SHAP Aggregation
Aggregate mean |SHAP| by modality. Expected breakdown:
- T2: 42.9%
- T1CE: 22.5%
- FLAIR: 21.1%
- T1: 13.5%

### 9.4 Cumulative SHAP Importance Curve
Plot features sorted by descending mean |SHAP| vs. cumulative % of total SHAP. The top ~210 features should capture nearly 100% of importance.

### 9.5 SHAP Waterfall Plots
Generate waterfall plots for the most confident correct class-0 and class-1 predictions. Compare against Fig. 6:
- Class-0 exemplar: f(x) = -1.687, E[f(x)] = 1.376
- Class-1 exemplar: f(x) = 2.987, E[f(x)] = 1.376

---

## Phase 10 — Reproducibility Checklist

- [ ] **Dataset version:** MU-Glioma-Post Version 1 from TCIA (DOI: 10.7937/7K9K-3C83)
- [ ] **Preprocessing:** N4 + z-score within lesion mask confirmed
- [ ] **PyRadiomics config:** Exact YAML matches authors' repo
- [ ] **Feature count:** 4,892 initial → 512 post-filter → 256 for best model
- [ ] **Patient-level test split:** Exactly 30 patients / 96 samples held out (43 class-0, 53 class-1)
- [ ] **No data leakage:** Feature ranking, CV folds, and calibration all performed without test patients
- [ ] **Optuna:** TPE sampler, 400 trials, patient-aware 5-fold GroupKFold
- [ ] **Calibration:** Platt scaling on OOF probabilities
- [ ] **AUC on test set:** ~0.80
- [ ] **Brier score improvement:** Raw ~0.093 → Calibrated ~0.088
- [ ] **SHAP explanations:** Top features match Table 3, modality breakdown matches Fig. 8

---

## Known Sources of Variance

Even following all steps exactly, small deviations may occur due to:

1. **Random seeds:** If the authors' exact seed for train/test split is unknown, the 30 held-out patients will differ, causing metric changes.
2. **Optuna stochasticity:** TPE sampling is stochastic; hyperparameters may differ across runs. Consider fixing the Optuna seed.
3. **LightGBM non-determinism:** Set `deterministic=True` and `force_row_wise=True` in LightGBM params for exact reproducibility (at the cost of speed).
4. **PyRadiomics version:** Minor version differences can change feature values. Pin the exact version from the authors' `requirements.txt`.
5. **Floating-point precision:** N4 and z-score operations may vary slightly across SimpleITK versions.

**Mitigation:** Clone the authors' repo first and use their exact seeds, config files, and package versions wherever available. If seeds are not documented, expect ±0.02 AUC variance on the test set.
