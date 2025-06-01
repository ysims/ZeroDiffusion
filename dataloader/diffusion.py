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

# This file contains the DiffusionDataset class, which is a PyTorch Dataset
# It generates samples using a diffusion model, typically for unseen classes in zero-shot learning

import torch
from torch.utils.data import Dataset


class DiffusionDataset(Dataset):
    def __init__(self, diffusion_model, feature_size, class_aux, num_samples):
        self.data = []
        self.labels = []

        # Get the device of the model
        device = next(diffusion_model.parameters()).device

        # Generate data for unseen classes
        for c in class_aux:
            for _ in range(num_samples):
                # Ensure c is a tensor of the same data type as the input tensor
                c_ = c.clone().detach().to(device).float()
                # Add noise to the class auxiliary vector
                aux_noise = c_ + torch.randn_like(c_) * 0.0
                # Generate a random sample and move it to the model's device
                # print dtype of inputs
                sample = diffusion_model(torch.randn(feature_size).to(device) * 0.1, aux_noise)
                self.data.append(sample)
                self.labels.append(c_)

        # Convert to tensor
        self.data = torch.stack(self.data)
        self.labels = torch.stack(self.labels)

        # Move data and labels to the model's device
        self.data = self.data.to(device)
        self.labels = self.labels.to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __add__(self, other):
        self.data = torch.cat((self.data, other.data))
        self.labels = torch.cat((self.labels, other.labels))
        return self
