# RobustKD

PyTorch implementation of reliability-weighted corruption-response distillation for lightweight image classification.

## Installation


Use Python 3.11 or newer. Create and activate a virtual environment, then install:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install:

```bash
python -m pip install -e .
``` For GPU execution, install matching PyTorch and torchvision builds using the [PyTorch selector](https://pytorch.org/get-started/locally/).


## Data and execution

`prepare` downloads [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html). Download and unpack [CIFAR-100-C](https://zenodo.org/records/3555552) separately under `data/CIFAR-100-C`; it must contain `labels.npy` and the 15 corruption arrays. Corruption benchmark images are used only by evaluation.

```bash
robustkd prepare
robustkd train --method teacher --output runs/teacher
robustkd train --method robustkd --teacher runs/teacher/model.pt --output runs/full
robustkd evaluate --method robustkd --checkpoint runs/full/model.pt --output runs/full
robustkd compare --inputs runs/full --output outputs/comparison
robustkd plot --output outputs/comparison
```

Run `train` and `evaluate` for `student`, `kd`, `feature`, `response` and `robustkd` in separate directories, using the same teacher for distillation variants. Supply all evaluated directories to `compare --inputs`. `response` disables only reliability weighting; `feature` retains clean logit/feature KD. `teacher` and `student` require no teacher checkpoint. To evaluate the teacher, select `--method teacher` and its checkpoint.

## Implementation

The teacher has three five-block residual stages of widths 64/128/256 (ResNet32x4); the student is torchvision ShuffleNetV2 x1.0 with stride-one stem and no maxpool for CIFAR. Feature losses use global pooled vectors and a training-only linear projection. The deployed checkpoint excludes teacher and projection. All student runs compute the same clean/corrupt views. The plain student is trained on clean labels; corrupt forwards preserve the matched training computation budget without adding a corrupted-label objective.

The objective combines clean CE, temperature-scaled KL, clean feature MSE, cosine logit-response distance and normalized feature-response distance. Teacher confidence times `exp(-eta*severity)` weights response terms. Online corruptions are Gaussian noise, horizontal motion blur, brightness and JPEG, sampled at continuous normalized severity; they are training transforms rather than exact CIFAR-C kernels. The other eleven benchmark corruptions form the unseen set.

`configs/default.json` contains the paper's 240-epoch SGD schedule, batch size 64, temperature 4, eta 1.5 and loss weights 1/1/0.5. `device`, `seed`, learning rate, milestones and loss weights are editable. `--data`, `--corruptions`, `--config`, `--teacher`, `--checkpoint` and `--output` select input/output locations. Each run saves the resolved settings, epoch training losses and final checkpoint. No test-based checkpoint selection is performed.

## Metrics and outputs

Evaluation computes clean accuracy; seen, unseen and overall accuracy over **all five severities**; per-corruption/severity accuracy; parameter count; and inference FLOPs. Conv/linear multiply and add count as two FLOPs; normalization, activation and pooling are excluded. The same convention applies to teacher and student. `corruptions.csv`, `severity.csv` and `metrics.json` contain generated measurements. Comparison writes CSV/JSON and plotting produces PNG/SVG. Rates use [0,1]. Keep seeds and teacher checkpoints paired across methods.

## Multiple seeds

Run matched configurations with different seeds in separate directories, then aggregate the generated metric files:

```bash
robustkd-report --inputs runs/seed0/metrics.json runs/seed1/metrics.json --output outputs/seeds
```

Use actual metric paths from your runs, including method subdirectories where applicable. Keep dataset, split and experimental condition fixed within a report. Summary CSV/JSON gives per-method arithmetic means and sample standard deviations; a single run has no standard-deviation estimate.
