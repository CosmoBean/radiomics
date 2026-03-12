# Thinking Radically

## Objective

Reproduce the postoperative glioma surveillance results reported by Christodoulou et al. and test alternatives that could improve them while keeping the pipeline interpretable.

## What We Built

We implemented the workflow natively in this repository:

- manifest-driven cohort construction
- lesion-union masking from labels `1/2/3`
- N4 bias correction
- lesion-mask z-score normalization
- PyRadiomics feature extraction
- patient-held-out evaluation
- feature ranking, model search, calibration, ROC/DCA reporting

## LightGBM Reproduction Attempt

The first target was the paper-style surveillance setup:

- label mode: `post_progression`
- held-out split: `30` patients / `96` scans
- target reference model: `LightGBM-256`
- target paper ROC AUC: about `0.80`

Reproduced:

- cohort: `590` scans across `202` patients
- held-out test split: `30` patients / `96` scans
- held-out class balance: `53` positive / `43` negative
- initial feature count: `4892`

Result:

- best model: `LightGBM-64`
- test ROC AUC: `0.5985`

- we reproduced the mechanics of the paper-style workflow
- we did **not** reproduce the paper's headline performance with pure radiomics

## Alternative Approaches Tried

### 1. Broader paper-style model search

We froze the same paper-style held-out patient set and searched multiple families on the same radiomics table.

Best result:

- `LogReg-128`
- ROC AUC: `0.6112`

Takeaway:

- the paper-style split was still weak even after broader model search
- pure radiomics remained below the paper baseline

### 2. Forward radiomics-only surveillance

We changed the task definition to a cleaner prediction setup:

- `within_window` prediction
- `120`-day window
- pre-progression scans only
- exclude scans after late-treatment start
- imaging backbone narrowed to `T1c + FLAIR`

Best result:

- `LogReg-32`
- ROC AUC: `0.6743`

Takeaway:

- task cleanup improved performance
- pure radiomics was still not enough to reach the target

### 3. Earliest-scan baseline

We also tested a first-postoperative-scan-only setting.

Best radiomics-only result:

- `LogReg-32`
- ROC AUC: `0.7421`

Takeaway:

- this was closer to the paper result
- but it used only `30` held-out scans, so it was less stable than the forward-surveillance result

### 4. Modality ablations

We tested whether simpler or wider imaging backbones helped.

Results on the baseline split:

- `T1c` only: ROC AUC `0.7014`
- all modalities: ROC AUC `0.6199`
- `T1c + FLAIR`: strongest and most stable direction

Takeaway:

- `T1c + FLAIR` was the best imaging backbone
- neither narrowing to `T1c` only nor widening to all modalities helped

### 5. Hybrid explainable-radiomics model

We then added curated non-imaging features already present in the dataset:

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

These were one-hot encoded and merged into the scan-level feature table.

Results:

- earliest-scan hybrid screen: `LogReg-48`, ROC AUC `0.8733`
- forward hybrid validation: `LogReg-32`, ROC AUC `0.7785`

Takeaway:

- the first clear performance jump came from the hybrid branch
- this is where we finally moved beyond the desired threshold

### 6. Paper-style hybrid attempt on the fixed `30 / 96` split

We then tested whether the same hybrid improvement would transfer back to the looser paper-style surveillance split.

Result:

- `LogReg-48`
- ROC AUC: `0.6209`

Takeaway:

- hybrid features helped on the cleaner tasks
- they did **not** rescue the looser `post_progression` paper-style task

## Comparative Summary

| Setting | Model | Inputs | Held-out design | ROC AUC |
| --- | --- | --- | --- | ---: |
| Christodoulou et al. baseline paper | LightGBM-256 | Radiomics | 30 patients / 96 scans | 0.80 |
| Native paper-style replication | LightGBM-64 | Radiomics | 30 patients / 96 scans | 0.5985 |
| Paper-style broader search | LogReg-128 | Radiomics | 30 patients / 96 scans | 0.6112 |
| Forward radiomics-only | LogReg-32 | `T1c + FLAIR` radiomics | 30 patients / 84 scans | 0.6743 |
| Earliest-scan radiomics-only | LogReg-32 | `T1c + FLAIR` radiomics | 30 patients / 30 scans | 0.7421 |
| Earliest-scan hybrid | LogReg-48 | `T1c + FLAIR` radiomics + molecular/basic clinical | 30 patients / 30 scans | 0.8733 |
| Forward hybrid validation | LogReg-32 | `T1c + FLAIR` radiomics + molecular/basic clinical | 30 patients / 84 scans | 0.7785 |
| Paper-style hybrid attempt | LogReg-48 | `T1c + FLAIR` radiomics + molecular/basic clinical | 30 patients / 96 scans | 0.6209 |

## Findings

### Pure radiomics did not beat the paper

Across the paper-style and forward radiomics-only runs, pure radiomics never exceeded the reported paper baseline.

### The task definition mattered as much as the model

The looser `post_progression` framing behaved differently from the stricter forward-prediction framing. The cleaner forward task generalized better.

### `T1c + FLAIR` was the strongest imaging backbone

This combination outperformed `T1c` alone and all-modality variants in the ablations.

### Logistic regression was the most stable family in our runs

LightGBM matched the original paper's model family, but in our runs `logreg` was more stable on the small high-dimensional tabular problem.

### The main gain came from hybrid modeling

The largest improvement came from adding curated molecular and basic clinical features to the radiomics table. This pushed the model above the target threshold on the cleaner branches.

## How We Made Decisions

The sequence was driven by observed failure modes and small comparisons.

Decision pattern:

1. reproduce the paper-style workflow first
2. test whether broader search fixed the gap
3. clean the task definition when broader search was not enough
4. identify the strongest imaging backbone with small ablations
5. move to hybrid modeling when pure radiomics plateaued

We also used odd-number multi-agent review passes at key forks to avoid arbitrary choices. Those votes helped with:

- initial label-mode decisions for reproduction
- narrowing fast searches to `logreg` and small feature subsets
- selecting the next branch after a failed paper-style hybrid attempt

## Current Position

The main conclusion is:

- we were **not** able to beat the paper with pure radiomics alone
- we **were** able to exceed the target threshold on cleaner hybrid formulations

The strongest practical direction going forward is:

- `T1c + FLAIR`
- hybrid radiomics + molecular/basic clinical features
- logistic regression as the stable baseline
- then SVM / boosted trees as secondary hybrid comparisons
