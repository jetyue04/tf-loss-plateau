# tf-loss-plateau

This is the code for the DSC180A project by Jet Yue with Prof. Tianhao Wang. The repo is based on the paper *"What Happens During the Loss Plateau? Understanding Abrupt Learning in Transformers"* (NeurIPS 2025) [arXiv: 2506.13688](https://arxiv.org/abs/2506.13688). This project extends on the paper by looking at the effect of task diversity.

## Overview

This project explores theoretical ML phenomena, focusing on loss plateaus in transformer models. It includes code to replicate results for algorithmic tasks such as **Moving Window Sum (MWS)** and **Moving Window Product (MWP)**.

## Project Structure

- `notebooks/` — Notebooks for testing and ablation studies  
- `src/` — Source code modules  
  - `legacy/` — Original code from the paper  
- `logs/` — Log files (ignored in git)  
- `wandb/` — Weights & Biases tracking directory (ignored in git)  
- `train.py` — Main training script  
- `env.yml` — Conda environment with dependencies  
- `.gitignore` — Specifies files/folders to ignore in git  
- `README.md` — This file  

## Setup

1. **Install dependencies**:
```bash
conda env create -f env.yml
conda activate emerge
```

2. Weights & Biases (optional): Add your W&B API key in train.py to track metrics.


### Run Experiment
Run the default experiment (Mixing MWS and MWP):
```bash
python train.py
```

Run with a custom config file:
```bash
python train.py --config path/to/config.yaml
```

### Data
No external data is required. All task data is generated programmatically based on the configuration.