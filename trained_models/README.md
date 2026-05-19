# Trained model checkpoints

This directory contains the trained checkpoints used for the machine-learning
results reported in the manuscript.

| File | Task | Output |
| --- | --- | --- |
| `gap_index_classifier_improved_best.pt` | Four-class gap-index classification | gap index category |
| `gap2_regressor_best.pt` | Gap-index-2 regression | lower and upper gap-edge frequencies |
| `gap3_regressor_best.pt` | Gap-index-3 regression | lower and upper gap-edge frequencies |

The two regressors are independent single-task models for the lower and upper
band-gap edges of the corresponding gap-index category.
