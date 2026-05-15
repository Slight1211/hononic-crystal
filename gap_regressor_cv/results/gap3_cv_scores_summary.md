# Gap3 Regressor CV Score Summary

All results are from five-fold cross-validation on a 20% subset of all gap-index-3 samples. The selection metric is the mean MAE of the lower and upper band-edge frequencies.

```text
available gap3 samples = 1,596,663
selected samples = 319,333
folds = 5
epochs per fold = 5
batch size = 256
```

| rank | config | mean edge MAE (kHz) | lower MAE (kHz) | upper MAE (kHz) | width MAE (kHz) | lower R2 | upper R2 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `lr0005` | 0.164867 | 0.078144 | 0.251590 | 0.260491 | 0.995967 | 0.997511 |
| 2 | `lr0004` | 0.176393 | 0.081674 | 0.271112 | 0.280921 | 0.995591 | 0.997129 |
| 3 | `dropout005` | 0.185710 | 0.082966 | 0.288453 | 0.292670 | 0.995685 | 0.996676 |
| 4 | `huber_beta05` | 0.189938 | 0.084537 | 0.295338 | 0.301685 | 0.995484 | 0.996518 |
| 5 | `layers6` | 0.198126 | 0.088505 | 0.307747 | 0.317303 | 0.994933 | 0.996315 |
| 6 | `width256` | 0.198661 | 0.101075 | 0.296247 | 0.311002 | 0.993553 | 0.996548 |
| 7 | `baseline_h224_l8` | 0.198857 | 0.091284 | 0.306431 | 0.318567 | 0.994478 | 0.996343 |
| 8 | `layers10` | 0.199406 | 0.088356 | 0.310457 | 0.317806 | 0.995131 | 0.996255 |
| 9 | `dropout015` | 0.206251 | 0.094954 | 0.317548 | 0.326000 | 0.994440 | 0.996067 |
| 10 | `width192` | 0.211285 | 0.096321 | 0.326250 | 0.335289 | 0.994230 | 0.995815 |
| 11 | `dropout020` | 0.216912 | 0.100877 | 0.332947 | 0.340196 | 0.993541 | 0.995761 |
| 12 | `lr0002` | 0.222564 | 0.097442 | 0.347687 | 0.356570 | 0.993917 | 0.995194 |

## Recommended Configuration

The selected gap3 configuration is:

```text
hidden_dim = 224
layers = 8
dropout = 0.10
lr = 5e-4
weight_decay = 1e-5
smooth_l1_beta = 1.0
```
