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
    def __init__(self, diffusion_model, feature_size, class_aux, num_samples, norm, generation_batch_size=128):
        self.data = []
        self.labels = []

        # Get the device of the model
        device = next(diffusion_model.parameters()).device

        # Determine if model has generate() method (DDPM) or uses forward (DenoiseNet)
        has_generate = hasattr(diffusion_model, 'generate') and callable(getattr(diffusion_model, 'generate'))
        generation_batch_size = max(1, int(generation_batch_size))

        # Generate data for unseen classes
        with torch.no_grad():
            for c in class_aux:
                # Ensure c is a tensor of the same data type as the input tensor
                c_ = c.clone().detach().to(device).float()
                generated_for_class = 0
                while generated_for_class < num_samples:
                    current_bs = min(generation_batch_size, num_samples - generated_for_class)

                    aux_batch = c_.unsqueeze(0).repeat(current_bs, 1)
                    noise = torch.randn(current_bs, feature_size, device=device)
                    noise = noise / (noise.norm(dim=1, keepdim=True) + 1e-8)
                    noise = noise * norm

                    # Use appropriate generation method
                    if has_generate:
                        # DDPM uses generate() method
                        samples = diffusion_model.generate(noise, aux_batch)
                    else:
                        # DenoiseNet uses forward() method
                        samples = diffusion_model(noise, aux_batch)

                    self.data.append(samples)
                    self.labels.append(aux_batch)
                    generated_for_class += current_bs

        # Store generated samples on CPU for efficient batched transfer during training.
        self.data = torch.cat(self.data, dim=0).detach().cpu()
        self.labels = torch.cat(self.labels, dim=0).detach().cpu()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __add__(self, other):
        self.data = torch.cat((self.data, other.data))
        self.labels = torch.cat((self.labels, other.labels))
        return self
