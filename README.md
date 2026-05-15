# Understanding Multimodal Learning: A Loss Landscape Smoothness Perspective

Official implementation of **Understanding Multimodal Learning: A Loss Landscape Smoothness Perspective**.

> Jae-Jun Lee, Sung Whan Yoon  
> **ICML 2026 Accepted**
>
> [ICML 2026] TBD

TL;DR: This paper studies multimodal learning through the lens of **loss landscape smoothness**. We show that multimodal learning can induce a convolutional smoothing effect over the loss landscape and propose **Distributional Multimodal Learning (DML)**, a simple stochastic modality-pairing strategy that further promotes flatter landscapes, robustness, and generalization.


## Installation

```bash
conda create -n DML python=3.9 -y
conda activate DML

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install timm transformers pandas scikit-learn
```

## Dataset Preparation

We use four multimodal benchmark datasets in our experiments:

| Dataset | Modalities (Official Dataset Link) | Description |
| --- | --- | --- |
| AVMNIST | Audio-Image ([Link](https://multibench.readthedocs.io/en/latest/start/datadownload.html#av-mnist)) | Digit classification dataset with paired audio and image modalities. |
| CREMA-D | Audio-Visual ([Link](https://github.com/CheyneyComputerScience/CREMA-D)) | Emotion recognition dataset based on speech and facial expressions. |
| Kinetics-Sounds | Audio-Visual ([Link](https://github.com/dancelogue/kinetics-datasets-downloader)) | Action recognition dataset derived from Kinetics, containing videos with audio cues. |
| UPMC-Food101 | Image-Text ([Link](https://www.kaggle.com/datasets/gianmarco96/upmcfood101)) | Multimodal food classification dataset with image and text modalities. |

We just followed the steps provided from above links to download and preprocess the datasets, especially for Kinetics-Sounds and CREMA-D.
<br>
For reproducibility, we provide our preprocessed version of all datasets used in our experiments.

$\Rightarrow$ **Preprocessed datasets (approximately 28GB):** [Google Drive Link](https://drive.google.com/file/d/1XJdf6LnfWDxX2TAHLimYblmIY3o8VY_t/view?usp=sharing)

After downloading, unzip it and set `--data_root` to the path of the unzipped directory when running the commands in the next section.

## Quick Start

### Training 

Commands for each dataset/modality (replace paths for your machine):

```bash
# CREMAD (Audio)
python main.py --mode train --data_root /path/to/CREMAD --dataset CREMAD --modality Audio --expt_dir ckpt --expt_name Audio --batch_size 16 --epochs 70 --learning_rate 1e-3 --optim sgd --weight_decay 5e-4 --lr_decay_step 30 --used_frames 1 --master_seed 1000 --num_exp 5 --save_checkpoint True

# KineticSound (Visual)
python main.py --mode train --data_root /path/to/kinetics_sound --dataset Kinetics --modality Visual --expt_dir ckpt --expt_name Visual --batch_size 64 --epochs 30 --learning_rate 1e-3 --optim adam --weight_decay 5e-4 --lr_decay_step 30 --used_frames 3 --master_seed 1000 --num_exp 5 --save_checkpoint True

# AVMNIST (Multimodal SML)
python main.py --mode train --data_root /path/to/AVMNIST --dataset AVMNIST --modality Multimodal --expt_dir ckpt --expt_name SML --batch_size 64 --epochs 70 --learning_rate 1e-3 --optim sgd --weight_decay 5e-4 --lr_decay_step 30 --master_seed 1000 --num_exp 5 --save_checkpoint True

# Food101 (Multimodal DML)
python main.py --mode train --data_root /path/to/Food101 --dataset Food101 --modality Multimodal --expt_dir ckpt --expt_name DML --batch_size 64 --epochs 70 --learning_rate 1e-2 --optim sgd --weight_decay 1e-3 --lr_decay_step 30 --num_workers 8 --shuffling --master_seed 1000 --num_exp 5 --save_checkpoint True
```

### Test only (no training):

```bash
python main.py \
  --mode test \
  --dataset CREMAD \
  --expt_name cremad_baseline \
  --master_seed 999 --num_exp 1 \
  --checkpoint_path checkpoint/CREMAD/cremad_baseline/999/ckpt_full_epoch_best.pth.tar
```

If `--checkpoint_path` is not provided in test mode, the runner tries:

```text
<expt_dir>/<dataset>/<expt_name>/<seed>/ckpt_full_epoch_best.pth.tar
```

## Project Layout

Current source layout is organized by responsibility:

- `run/`: dataset-specific train/test execution entrypoints
- `tasks/`: model + optimizer + dataloader wiring
- `model/`: model definitions
- `dataloader/`: dataset loaders and preprocessing
- `utils/`: config, logging, metrics, optimizer helpers

## Multi-Run Command Template

Generalized version of the commands you ran (replace paths/CPU lists for your server):

```bash
CPU_LISTS=("21-28" "1-8" "41-48" "61-68")

DATA_ROOT_CREMAD=/path/to/CREMAD
DATA_ROOT_KINETIC=/path/to/kinetics_sound
DATA_ROOT_AVMNIST=/path/to/AVMNIST
DATA_ROOT_FOOD101=/path/to/Food101

# KineticSound
CUDA_VISIBLE_DEVICES=0 taskset --cpu-list ${CPU_LISTS[0]} python main.py --data_root ${DATA_ROOT_KINETIC} --modality Visual --master_seed 1000 --num_exp 5 --expt_dir ckpt --expt_name Visual --batch_size 64 --epochs 30 --learning_rate 1e-3 --dataset Kinetics --save_checkpoint True --optim adam --used_frames 3 --weight_decay 5e-4 --lr_decay_step 30 &
CUDA_VISIBLE_DEVICES=1 taskset --cpu-list ${CPU_LISTS[1]} python main.py --data_root ${DATA_ROOT_KINETIC} --modality Audio --master_seed 1000 --num_exp 5 --expt_dir ckpt --expt_name Audio --batch_size 64 --epochs 30 --learning_rate 1e-3 --dataset Kinetics --save_checkpoint True --optim adam --used_frames 3 --weight_decay 5e-4 --lr_decay_step 30 &
CUDA_VISIBLE_DEVICES=2 taskset --cpu-list ${CPU_LISTS[2]} python main.py --data_root ${DATA_ROOT_KINETIC} --modality Multimodal --master_seed 1000 --num_exp 5 --expt_dir ckpt --expt_name DML --batch_size 64 --epochs 30 --learning_rate 1e-3 --dataset Kinetics --save_checkpoint True --optim adam --used_frames 3 --weight_decay 5e-4 --lr_decay_step 30 --shuffling &
CUDA_VISIBLE_DEVICES=3 taskset --cpu-list ${CPU_LISTS[3]} python main.py --data_root ${DATA_ROOT_KINETIC} --modality Multimodal --master_seed 1000 --num_exp 5 --expt_dir ckpt --expt_name SML --batch_size 64 --epochs 30 --learning_rate 1e-3 --dataset Kinetics --save_checkpoint True --optim adam --used_frames 3 --weight_decay 5e-4 --lr_decay_step 30 &
wait

# AVMNIST
CUDA_VISIBLE_DEVICES=0 taskset --cpu-list ${CPU_LISTS[0]} python main.py --data_root ${DATA_ROOT_AVMNIST} --modality Visual --master_seed 1000 --num_exp 5 --expt_dir ckpt --expt_name Image --batch_size 64 --epochs 70 --learning_rate 1e-3 --weight_decay 5e-4 --dataset AVMNIST --save_checkpoint True --optim sgd --lr_decay_step 30 &
CUDA_VISIBLE_DEVICES=1 taskset --cpu-list ${CPU_LISTS[1]} python main.py --data_root ${DATA_ROOT_AVMNIST} --modality Audio --master_seed 1000 --num_exp 5 --expt_dir ckpt --expt_name Audio --batch_size 64 --epochs 70 --learning_rate 1e-3 --weight_decay 5e-4 --dataset AVMNIST --save_checkpoint True --optim sgd --lr_decay_step 30 &
CUDA_VISIBLE_DEVICES=2 taskset --cpu-list ${CPU_LISTS[2]} python main.py --data_root ${DATA_ROOT_AVMNIST} --modality Multimodal --master_seed 1000 --num_exp 5 --expt_dir ckpt --expt_name SML --batch_size 64 --epochs 70 --learning_rate 1e-3 --weight_decay 5e-4 --dataset AVMNIST --save_checkpoint True --optim sgd --lr_decay_step 30 &
CUDA_VISIBLE_DEVICES=3 taskset --cpu-list ${CPU_LISTS[3]} python main.py --data_root ${DATA_ROOT_AVMNIST} --modality Multimodal --master_seed 1000 --num_exp 5 --expt_dir ckpt --expt_name DML --batch_size 64 --epochs 70 --learning_rate 1e-3 --weight_decay 5e-4 --dataset AVMNIST --save_checkpoint True --optim sgd --lr_decay_step 30 --shuffling &
wait

# CREMAD
CUDA_VISIBLE_DEVICES=0 taskset --cpu-list ${CPU_LISTS[0]} python main.py --data_root ${DATA_ROOT_CREMAD} --modality Audio --master_seed 1000 --num_exp 5 --expt_dir ckpt --expt_name Audio --dataset CREMAD --save_checkpoint True --batch_size 16 --epochs 70 --learning_rate 1e-3 --optim sgd --weight_decay 5e-4 --lr_decay_step 30 --used_frames 1 &
CUDA_VISIBLE_DEVICES=1 taskset --cpu-list ${CPU_LISTS[1]} python main.py --data_root ${DATA_ROOT_CREMAD} --modality Visual --master_seed 1000 --num_exp 5 --expt_dir ckpt --expt_name Visual --dataset CREMAD --save_checkpoint True --batch_size 16 --epochs 70 --learning_rate 1e-3 --optim sgd --weight_decay 5e-4 --lr_decay_step 30 --used_frames 1 &
CUDA_VISIBLE_DEVICES=2 taskset --cpu-list ${CPU_LISTS[2]} python main.py --data_root ${DATA_ROOT_CREMAD} --modality Multimodal --master_seed 1000 --num_exp 5 --expt_dir ckpt --expt_name DML --dataset CREMAD --save_checkpoint True --batch_size 16 --epochs 70 --learning_rate 1e-3 --optim sgd --weight_decay 5e-4 --lr_decay_step 30 --used_frames 1 --shuffling &
CUDA_VISIBLE_DEVICES=3 taskset --cpu-list ${CPU_LISTS[3]} python main.py --data_root ${DATA_ROOT_CREMAD} --modality Multimodal --master_seed 1000 --num_exp 5 --expt_dir ckpt --expt_name SML --dataset CREMAD --save_checkpoint True --batch_size 16 --epochs 70 --learning_rate 1e-3 --optim sgd --weight_decay 5e-4 --lr_decay_step 50 --used_frames 1 &
wait

# Food101
CUDA_VISIBLE_DEVICES=0 taskset --cpu-list ${CPU_LISTS[0]} python main.py --data_root ${DATA_ROOT_FOOD101} --modality Visual --master_seed 1000 --num_exp 5 --expt_dir ckpt --expt_name Visual --batch_size 64 --epochs 70 --learning_rate 1e-2 --dataset Food101 --save_checkpoint True --optim sgd --weight_decay 1e-4 --lr_decay_step 30 --num_workers 8 &
CUDA_VISIBLE_DEVICES=1 taskset --cpu-list ${CPU_LISTS[1]} python main.py --data_root ${DATA_ROOT_FOOD101} --modality Text --master_seed 1000 --num_exp 5 --expt_dir ckpt --expt_name Text --batch_size 64 --epochs 70 --learning_rate 1e-3 --dataset Food101 --save_checkpoint True --optim sgd --weight_decay 1e-3 --lr_decay_step 30 --num_workers 8 &
CUDA_VISIBLE_DEVICES=2 taskset --cpu-list ${CPU_LISTS[2]} python main.py --data_root ${DATA_ROOT_FOOD101} --modality Multimodal --master_seed 1000 --num_exp 5 --expt_dir ckpt --expt_name SML --batch_size 64 --epochs 70 --learning_rate 1e-2 --dataset Food101 --save_checkpoint True --optim sgd --weight_decay 1e-3 --lr_decay_step 30 --num_workers 8 &
CUDA_VISIBLE_DEVICES=3 taskset --cpu-list ${CPU_LISTS[3]} python main.py --data_root ${DATA_ROOT_FOOD101} --modality Multimodal --master_seed 1000 --num_exp 5 --expt_dir ckpt --expt_name DML --batch_size 64 --epochs 70 --learning_rate 1e-2 --dataset Food101 --save_checkpoint True --optim sgd --weight_decay 1e-3 --lr_decay_step 30 --shuffling --num_workers 8 &
wait
```

## Analysis

We provide the scripts for all analyses in the `analysis/` directory. You can run them after training finishes and checkpoints are saved.
Read `README.md` at `analysis/` for more details on how to run each analysis script.

## Citation

Soon to be updated with the official citation information after the camera-ready version is available.