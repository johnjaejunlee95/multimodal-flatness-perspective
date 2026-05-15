# Analysis

Unified analysis workflows for:
- adversarial robustness
- flatness
- generalization (LPF)
- Hessian max-eigenvalue

All migrated scripts use a shared manifest:
- `analysis/configs/checkpoints.yaml`

## Directory Layout

```text
analysis/
├── configs/
│   └── checkpoints.yaml
├── scripts/
│   ├── run_adversarial.sh
│   ├── run_flatness.sh
│   ├── run_generalization.sh
│   └── run_hessian.sh
├── src/
│   ├── adversarial/
│   ├── flatness/
│   ├── generalization/
│   ├── hessian/
│   └── common/
└── results/
```

## Requirements

From repo root:

```bash
pip install pyyaml
```

## Quick Start

From repo root:

```bash
bash analysis/scripts/run_adversarial.sh all
bash analysis/scripts/run_flatness.sh all
bash analysis/scripts/run_generalization.sh all
bash analysis/scripts/run_hessian.sh all
```

Run a single dataset:

```bash
bash analysis/scripts/run_hessian.sh Food101
bash analysis/scripts/run_generalization.sh kinetic
```

## Direct Script Usage

Every migrated script supports:
- `--manifest` (default: `analysis/configs/checkpoints.yaml`)
- `--profile`
- `--dataset`
- `--ckpt-root-override`

Examples:

```bash
python analysis/src/adversarial/adversarial_robustness_avmnist.py \
  --dataset AVMNIST \
  --manifest analysis/configs/checkpoints.yaml \
  --profile adversarial_avmnist_default

python analysis/src/flatness/flatness_multimodal_cremad.py \
  --dataset CREMAD \
  --manifest analysis/configs/checkpoints.yaml \
  --profile flatness_cremad_default

python analysis/src/generalization/generalization_measure_ks.py \
  --dataset kinetic \
  --manifest analysis/configs/checkpoints.yaml \
  --profile generalization_kineticsound_default
```

DDP Hessian example:

```bash
torchrun --nproc_per_node=4 analysis/src/hessian/hessian_max_eigenvalues_food101.py \
  --dataset Food101 \
  --manifest analysis/configs/checkpoints.yaml \
  --profile hessian_food101_default
```

## Manifest Notes

- Checkpoint path resolution is centralized in `analysis/configs/checkpoints.yaml`.
- Profiles define dataset-specific checkpoint variants and seeds.
- `--ckpt-root-override` lets you remap the project root without editing YAML.

## Outputs

- Generated artifacts are written under `analysis/results/` (or script-specific `results/...` paths already used by each script).
