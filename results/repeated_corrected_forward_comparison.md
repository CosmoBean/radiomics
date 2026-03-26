# Repeated Corrected Forward Evaluation Checkpoint

Date: 2026-03-26

## Completed Seeds

| Seed | `hybrid_basic` ROC AUC | `report_full` ROC AUC |
| --- | ---: | ---: |
| 42 | 0.7785 | 0.7743 |
| 52 | 0.6262 | 0.6407 |
| 62 | 0.8043 | 0.7918 |
| 72 | 0.6341 | pending |
| 82 | 0.5819 | pending |

## Current Aggregates

- `hybrid_basic` across 5 corrected seeds: mean ROC AUC `0.6850 +/- 0.0995`
- `report_full` across 3 corrected seeds: mean ROC AUC `0.7356 +/- 0.0827`
- On the matched completed seeds `42/52/62`:
  - `hybrid_basic` mean ROC AUC `0.7363`
  - `report_full` mean ROC AUC `0.7356`
  - mean difference (`report_full - hybrid_basic`): `-0.0007`

## Takeaway

With the corrected repeated-split evaluation, the engineered report feature set is currently performing about the same as the existing `hybrid_basic` baseline on matched seeds, not clearly better.
