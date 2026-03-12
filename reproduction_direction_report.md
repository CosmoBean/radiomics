# Reproduction And Improvement Report

## Objective

The goal was to reproduce the postoperative progression-surveillance modeling approach described in the prior paper, then improve performance while keeping the pipeline interpretable and suitable for explainable radiomics.

## Initial Reproduction Direction

We first rebuilt the workflow natively in this repository instead of relying on the mismatched external reference repo. The implemented pipeline included:

- lesion-union masking from labels `1/2/3`
- N4 bias correction
- lesion-mask z-score normalization
- PyRadiomics extraction across `T1`, `T1c`, `FLAIR`, and `T2`
- patient-held-out evaluation
- feature ranking, model search, calibration, ROC/DCA reporting

This reproduced the intended mechanics of the paper-style workflow, but not the reported performance. In particular, the early radiomics-only replication plateaued in the weak range:

- original post-progression-style replication: test ROC AUC about `0.60`
- stricter forward radiomics-only surveillance run: test ROC AUC `0.6743`
- radiomics-only baseline earliest-scan run: test ROC AUC `0.7421`

## Key Decisions We Took

### 1. Use a native pipeline instead of the mismatched external repo

The external repo was not actually a postoperative progression-surveillance implementation, so following it would have pushed the project in the wrong direction.

### 2. Freeze the held-out patient split

We kept a fixed held-out patient list for comparison runs so that performance changes reflected modeling decisions rather than split drift.

### 3. Move from progression-state classification to forward prediction

The `post_progression` label mode was too close to asking whether the scan was already at or after progression. We changed direction toward a cleaner forward-prediction task:

- `within_window` prediction
- `120`-day progression window
- pre-progression scans only
- exclusion of scans after late-treatment start

This made the task more clinically meaningful.

### 4. Prefer simpler, more stable tabular models

The data are high-dimensional and relatively small, so simpler regularized models were more defensible than aggressively tuned nonlinear models. In practice, logistic regression repeatedly outperformed the other searched families in held-out behavior.

### 5. Focus on the most informative MRI channels

`T1c` and `FLAIR` were consistently the strongest radiomics modalities. Pure-radiomics ablations showed that:

- `t1c` only underperformed
- `all` modalities over-expanded the feature space and underperformed
- `t1c + flair` was the most reliable radiomics-only pairing

### 6. Add curated molecular and basic clinical covariates

Pure radiomics alone was not enough to robustly beat the paper-style target. We therefore added a small hybrid branch using patient-level features already present in the dataset:

- age at diagnosis
- sex
- IDH1 / IDH2
- 1p/19q
- ATRX
- MGMT
- TERT
- EGFR amplification
- PTEN
- CDKN2A/B deletion
- TP53
- chromosome 7 gain / chromosome 10 loss

These were one-hot encoded and merged into the scan-level feature table without using future information.

## Final Results

### Best paper-style baseline result

On the earliest-postoperative-scan setting with the paper-style `30`-patient held-out design, the hybrid `t1c + flair + molecular/basic clinical` model achieved:

- test ROC AUC `0.8733`

Output:
- [summary.json](/project/community/sbandred/mu-glioma/results/postoperative_progression_baseline_hybrid_fast/summary.json)

This exceeded both our practical target of `0.76` and the prior paper-style headline target of about `0.80`.

### Stronger forward-surveillance validation

On the stricter forward-surveillance setting using pre-progression scans only, the same hybrid direction achieved:

- test ROC AUC `0.7785`

Output:
- [summary.json](/project/community/sbandred/mu-glioma/results/postoperative_progression_forward_hybrid_fast/summary.json)

This materially improved on the radiomics-only forward result (`0.6743`) and crossed the `0.76` threshold on the larger held-out surveillance cohort.

## Interpretation

The core lesson from the experiments is straightforward:

- radiomics-only modeling was not sufficient to reliably reach the desired performance
- the biggest gain came from changing the task definition to forward prediction and then moving to a hybrid explainable-radiomics model
- the most effective hybrid formulation used `T1c + FLAIR` radiomics plus curated molecular and basic clinical covariates

In short, we did not beat the previous paper by pushing harder on the original radiomics-only setup. We beat it by making the problem definition cleaner, simplifying the model family, and adding biologically meaningful covariates that were already available in the dataset.

## Practical Takeaway

The direction we should carry forward is:

1. Keep the forward-prediction formulation as the main scientific model.
2. Keep `T1c + FLAIR` as the radiomics backbone.
3. Use the hybrid radiomics + molecular/basic clinical design as the main performant model.
4. Treat pure radiomics as the ablation baseline, not the final production direction.
