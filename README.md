# RobustKD

Implementation of reliability-weighted corruption-response distillation for lightweight image classification.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

`prepare` downloads [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html). Download [CIFAR-100-C](https://zenodo.org/records/3555552) separately to `data/CIFAR-100-C`, containing `labels.npy` and corruption arrays; use `--corruptions` to change that location.

```bash
robustkd prepare
```

## Training and execution

`configs/default.json` records seed `0`. Keep it fixed across paired methods and use separate output directories.

```bash
robustkd train --method teacher --output runs/teacher
robustkd train --method robustkd --teacher runs/teacher/model.pt --output runs/full
```

## Evaluation

```bash
robustkd evaluate --method robustkd --checkpoint runs/full/model.pt --output runs/full
```

## Comparisons and ablations

`--method` selects the implemented method or component control. Train separate checkpoints where applicable. `compare --inputs` aggregates evaluated run directories.

```bash
robustkd compare --inputs runs/full --output outputs/comparison
robustkd plot --output outputs/comparison
```

## Outputs

The explicit run directories contain `model.pt`, `corruptions.csv`, `metrics.json`, `severity.csv`, `history.csv`. Comparisons write CSV/JSON and plots derived from those saved measurements.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, torchvision 0.22.1, scipy 1.15.3, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
