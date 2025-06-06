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

# =============================================================================

# This file contains the ESC50Dataset class, which is a PyTorch Dataset, and a function to create ESC50 datasets
# The function returns validation and training datasets, and some dataset parameters


import numpy as np
import pickle
import sys
import torch
from concurrent.futures import ThreadPoolExecutor

np.set_printoptions(threshold=sys.maxsize)


def process_sample(i, all_labels, all_features, all_auxiliary, split, val_classes, test_classes):
    """Process a single sample and classify it into train or validation."""
    label = all_labels[i]
    feature = all_features[i]
    auxiliary = all_auxiliary[i]

    if split == "test":
        if label in test_classes:
            return "val", label, feature, auxiliary
        else:
            return "train", label, feature, auxiliary
    else:
        if label in test_classes:
            return None  # Skip test classes
        elif label in val_classes:
            return "val", label, feature, auxiliary
        else:
            return "train", label, feature, auxiliary


def create_esc50_datasets(file, split, val_classes, test_classes, device):
    with open(file, "rb") as f:
        data = pickle.load(f)

    all_labels = np.array(data["labels"])
    all_features = np.array([list(d.to("cpu")[0]) for d in data["features"]])
    all_auxiliary = np.array(data["auxiliary"])

    train_labels, train_features, train_auxiliary = [], [], []
    val_labels, val_features, val_auxiliary = [], [], []

    # Parallelize the loop using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(
                process_sample,
                range(len(all_labels)),
                [all_labels] * len(all_labels),
                [all_features] * len(all_labels),
                [all_auxiliary] * len(all_labels),
                [split] * len(all_labels),
                [val_classes] * len(all_labels),
                [test_classes] * len(all_labels),
            )
        )

    # Collect results
    for result in results:
        if result is None:
            continue
        category, label, feature, auxiliary = result
        if category == "train":
            train_labels.append(label)
            train_features.append(feature)
            train_auxiliary.append(auxiliary)
        elif category == "val":
            val_labels.append(label)
            val_features.append(feature)
            val_auxiliary.append(auxiliary)

    # Convert to tensors and remove duplicates
    unique_val_auxiliary = torch.tensor(
        np.unique([tuple(a) for a in val_auxiliary], axis=0)
    ).to(device)
    unique_train_auxiliary = torch.tensor(
        np.unique([tuple(a) for a in train_auxiliary], axis=0)
    ).to(device)

    train_dataset = ESC50Dataset(
        torch.tensor(train_labels).float().to(device),
        torch.tensor(train_features).float().to(device),
        torch.tensor(train_auxiliary).float().to(device),
    )
    val_dataset = ESC50Dataset(
        torch.tensor(val_labels).float().to(device),
        torch.tensor(val_features).float().to(device),
        torch.tensor(val_auxiliary).float().to(device),
    )

    dataset_params = {
        "aux_dim": train_auxiliary[0].shape[0],
        "feat_dim": train_features[0].shape[0],
        "val_auxiliary": unique_val_auxiliary,
        "train_auxiliary": unique_train_auxiliary,
    }

    return train_dataset, val_dataset, dataset_params


class ESC50Dataset(torch.utils.data.Dataset):
    def __init__(self, labels, features, auxiliary):
        self.labels = labels
        self.features = features
        self.auxiliary = auxiliary

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.labels[idx],
            self.features[idx],
            self.auxiliary[idx],
        )