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

class DAEInterp(nn.Module):
    """
    Denoising Autoencoder with Interpolation-based noise.
    """
    
    def __init__(self, input_dim, hidden_dim, aux_dim, n_layers=1, dropout=0.3, 
                 activation="leakyrelu", use_residual=False, use_layernorm=False,
                 output_activation="tanh"):
        super(DAEInterp, self).__init__()
        
        self.input_dim = input_dim
        self.use_residual = use_residual
        
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim + aux_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(self._make_activation(activation))
        layers.append(nn.Dropout(dropout))
        
        # Additional hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._make_activation(activation))
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        # Output activation
        if output_activation == "tanh":
            layers.append(nn.Tanh())
        elif output_activation == "none":
            pass  # No activation
        elif output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        
        self.layers = nn.Sequential(*layers)
        
        # For residual connection, need to project input to output dim
        if use_residual:
            self.residual_proj = nn.Linear(input_dim + aux_dim, input_dim)
    
    def _make_activation(self, activation):
        return {
            "leakyrelu": nn.LeakyReLU(inplace=True),
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(inplace=True),
            "elu": nn.ELU(inplace=True),
        }.get(activation, nn.LeakyReLU(inplace=True))

    def forward(self, x, aux):
        x = x if x.dim() > 1 else x.unsqueeze(0)
        aux = aux if aux.dim() > 1 else aux.unsqueeze(0)
        combined = torch.cat((x, aux), dim=1)
        out = self.layers(combined)
        
        if self.use_residual:
            out = out + self.residual_proj(combined)
        
        return out

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


# Alias for backwards compatibility
Diffusion = DAEInterp