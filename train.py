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

# The entry point for the program to train a model for the ZeroDiffusion method.
# Runs ten random seeds of repeated training.
# Training parameters are found here. This dictionary of parameters can be used to 
# perform a grid search over the hyperparameters.

import torch
import random
import numpy as np
import statistics

from setup import setup
from train_diffusion import train
from dataloader.esc50 import create_esc50_datasets

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

# All the tunable parameters for training
config = {
    "diffusion_lr": 1e-3,
    "diffusion_batch_size": 64,
    "diffusion_hidden_dim": 128,
    "diffusion_epoch": 100,
    "classifier_hidden_dim": 128,
    "classifier_learning_rate": 1e-4,
    "classifier_dataset_size": params.cls_dataset_size,  # this is per-class
    "classifier_batch_size": 64,
    "classifier_epoch": 20,
}# T1 has cls batch 64
# 1e-3 16 128 100 64 1e-3 1440 52 10

seeds = [
    random.randrange(0, 9999999) for _ in range(0, 10)
]  # <- Train several times randomly
n_trials = len(seeds)

print("Running {} trials.".format(n_trials))

accs = []

for trial, seed in enumerate(seeds):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Train the model
    accs.append(train(config, fixed_config)["mean_accuracy"])


print("Mean acc is", statistics.mean(accs))
print("Standard deviation acc is", statistics.stdev(accs))
print("Best acc is {}".format(max(accs)))

print("Experiment was type {} : folder {}".format(params.split, params.data))
