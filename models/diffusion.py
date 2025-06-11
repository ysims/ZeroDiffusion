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

# This file contains the Diffusion class, which is a PyTorch artificial neural network module
# It learns to reconstruct the input data with added noise

import torch
from torch import nn

class Diffusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, aux_dim):
        super(Diffusion, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim + aux_dim, hidden_dim),
            # nn.LeakyReLU(0.01),
            # nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )

    def forward(self, x, aux):
        # Ensure x and aux are 2D tensors
        x = x if x.dim() > 1 else x.unsqueeze(0)
        aux = aux if aux.dim() > 1 else aux.unsqueeze(0)

        # Jitter aux
        aux = aux + torch.randn_like(aux) * 0.1

        x = torch.cat((x, aux), dim=1)  # concatenate x and aux
        x = self.layers(x)
        return x

    # Distort the input data by interpolating between x and a noise vector of the same norm
    def distort(self, x, epoch_percentage):
        # Generate random noise
        noise = torch.randn_like(x)
        # Compute norms
        x_norm = x.norm(dim=1, keepdim=True)
        noise_norm = noise.norm(dim=1, keepdim=True)
        # Scale noise to have the same norm as x
        scaled_noise = noise / (noise_norm + 1e-8) * x_norm
        # Interpolate between x and scaled_noise
        return (1 - epoch_percentage) * x + epoch_percentage * scaled_noise