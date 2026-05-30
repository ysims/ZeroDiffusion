# MIT License

# Copyright (c) 2024 Ysobel Sims

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ==============================================================================

# Entry point for DDPM training on ZeroDiffusion.
# Runs ten random seeds of repeated training with DDPM diffusion model.

import torch
import random
import numpy as np
import statistics
import argparse

from setup import setup
from train_denoise import train
from dataloader.esc50 import create_esc50_datasets

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train DDPM diffusion model for zero-shot learning')
    parser.add_argument('data', help='Path to data pickle')
    parser.add_argument('--dataset', default='ESC-50', help='Dataset name')
    parser.add_argument('--split', default='fold0', help='Data split')
    parser.add_argument('--val_classes', type=int, default=10, help='Number of validation classes')
    parser.add_argument('--test_classes', type=int, default=10, help='Number of test classes')
    parser.add_argument('--cls_dataset_size', type=int, default=40, help='Number of synthetic samples per class')
    parser.add_argument('--device', default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Get non-tuning params
    params = setup()

    # SET UP TEST, VALIDATION AND TRAINING DATASETS
    train_set, val_set, dataset_params = create_esc50_datasets(
        params.data, params.split, params.val_classes, params.test_classes, params.device
    )

    # Training parameters that don't need tuning
    fixed_config = {
        "device": params.device,
        "auxiliary_dim": dataset_params["aux_dim"],
        "feature_dim": dataset_params["feat_dim"],
        "val_set": val_set,
        "train_set": train_set,
        "val_auxiliary": dataset_params["val_auxiliary"],
        "train_auxiliary": dataset_params["train_auxiliary"],
    }

    # All the tunable parameters for DDPM training
    config = {
        "diffusion_lr": 1e-5,
        "diffusion_batch_size": 128,
        "diffusion_hidden_dim": 64,
        "diffusion_epoch": 200,
        "ddpm_n_layers": 1,
        "ddpm_n_timesteps": 1000,
        "ddpm_dropout": 0.0,
        "ddpm_use_layernorm": True,
        "classifier_hidden_dim": 128,
        "classifier_learning_rate": 1e-4,
        "classifier_dataset_size": params.cls_dataset_size,  # this is per-class
        "generation_batch_size": 256,
        "classifier_batch_size": 16,
        "classifier_epoch": 20,
    }

    seeds = [
        random.randrange(0, 9999999) for _ in range(0, 10)
    ]  # <- Train several times randomly
    n_trials = len(seeds)

    print("Running {} trials with DDPM method.".format(n_trials))

    accs = []

    for trial, seed in enumerate(seeds):
        print(f"\n=== Trial {trial + 1}/{n_trials} with seed {seed} ===")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Train the model with DDPM
        accs.append(train(config, fixed_config, method="ddpm")["mean_accuracy"])


    print("\n=== DDPM Results ===")
    print("Mean acc is", statistics.mean(accs))
    print("Standard deviation acc is", statistics.stdev(accs))
    print("Best acc is {}".format(max(accs)))

    print("Experiment was type {} : folder {}".format(params.split, params.data))

if __name__ == "__main__":
    main()
