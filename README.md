# ZeroDiffusion

ZeroDiffusion — a zero-shot learning diffusion method for audio classification and generation experiments.

This repository contains training and evaluation scripts that rely on precomputed audio embeddings. The workflow is:

- Train or obtain an audio embedding model (see data/audio_embeddings/README.md).
- Precompute embeddings and save them as a pickle in the `data/` folder (see `data/README.md`).
- Run the ZeroDiffusion training scripts.

Prerequisites

- Python 3.8+ (use a virtual environment)
- Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Quick start

1. Prepare audio embeddings and place the dataset pickle under `data/`.
2. Train the model (example):

```bash
python train.py ../data/ESC_50_synonyms/fold04.pickle --dataset ESC-50 --split fold0
```

There are additional training/evaluation scripts in the repo:

- `train.py` — primary ZeroDiffusion training entrypoint.
- `train_ddpm.py`, `train_denoise.py` — DDPM / denoising variants.
- `run.sh`, `run_ddpm.sh` — convenience wrappers for common runs.

Repository layout

- `data/` — dataset helpers, embedding pickles, and dataset README.
- `dataloader/` — data loading and diffusion-related dataloaders.
- `models/` — model implementations (classifier, ddpm, denoise).
- `results/` — generated logs and outputs from experiments.

Notes

- See `data/README.md` and `audio_embeddings/README.md` for dataset and embedding preparation details.
- Adjust hyperparameters in the training scripts or pass flags on the command line.

If you'd like, I can also:

- Add example commands for the other scripts.
- Add a minimal config file for common hyperparameters.
- Create a short tutorial notebook demonstrating end-to-end runs.
