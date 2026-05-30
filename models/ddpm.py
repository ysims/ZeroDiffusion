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

# This file contains the DDPM (Denoising Diffusion Probabilistic Models) class
# It implements the full DDPM framework with linear noise scheduling

import math
import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for timesteps"""
    
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Models (DDPM)
    
    A diffusion-based generative model that learns to reverse a forward diffusion process
    by predicting noise added at each timestep.
    """
    
    def __init__(self, input_dim, aux_dim, hidden_dim, n_layers=4, 
                 n_timesteps=1000, dropout=0.3, activation="leakyrelu",
                 use_layernorm=True):
        super(DDPM, self).__init__()
        
        self.input_dim = input_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_timesteps = n_timesteps
        
        # Initialize noise schedule (linear)
        self.register_buffer('betas', self._linear_beta_schedule())
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', 
                           torch.cat([torch.ones(1), self.alphas_cumprod[:-1]]))
        
        # Precompute useful values
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                           torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', 
                           torch.sqrt(1.0 / self.alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', 
                           torch.sqrt(1.0 / self.alphas_cumprod - 1.0))
        
        # Posterior variance
        self.register_buffer('posterior_variance',
                           self.betas * (1.0 - self.alphas_cumprod_prev) / 
                           (1.0 - self.alphas_cumprod))
        
        # Timestep embedding
        emb_dim = hidden_dim * 2
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, emb_dim),
            self._make_activation(activation),
            nn.Linear(emb_dim, emb_dim),
        )
        
        # Auxiliary embedding
        self.aux_emb = nn.Sequential(
            nn.Linear(aux_dim, hidden_dim),
            self._make_activation(activation),
            nn.Linear(hidden_dim, emb_dim),
        )
        
        # Main network layers
        layers = []
        
        # First layer: projects concatenated input + embeddings to hidden
        layers.append(nn.Linear(input_dim + 2 * emb_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(self._make_activation(activation))
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._make_activation(activation))
            layers.append(nn.Dropout(dropout))
        
        # Output layer: predict noise (same dim as input)
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.layers = nn.Sequential(*layers)
    
    def _linear_beta_schedule(self):
        """Linear schedule for betas"""
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, self.n_timesteps)
    
    def _make_activation(self, activation):
        return {
            "leakyrelu": nn.LeakyReLU(inplace=True),
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(inplace=True),
            "elu": nn.ELU(inplace=True),
        }.get(activation, nn.LeakyReLU(inplace=True))
    
    def forward(self, x, t, aux):
        """
        Forward pass to predict noise.
        
        Args:
            x: noisy samples [batch_size, input_dim]
            t: timestep indices [batch_size]
            aux: auxiliary information [batch_size, aux_dim]
        
        Returns:
            predicted noise [batch_size, input_dim]
        """
        # Ensure correct shapes
        x = x if x.dim() > 1 else x.unsqueeze(0)
        aux = aux if aux.dim() > 1 else aux.unsqueeze(0)
        
        # Get embeddings
        t_emb = self.time_emb(t)
        aux_emb = self.aux_emb(aux)
        
        # Concatenate input with embeddings and process
        combined = torch.cat([x, t_emb, aux_emb], dim=1)
        noise_pred = self.layers(combined)
        
        return noise_pred
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: add noise to x_0 at timestep t.
        
        Args:
            x_0: original clean samples [batch_size, input_dim]
            t: timestep indices [batch_size]
            noise: optional pre-generated noise
        
        Returns:
            noisy sample x_t, and the noise used
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        if sqrt_alpha.dim() == 1:
            sqrt_alpha = sqrt_alpha.unsqueeze(1) if x_0.dim() > 1 else sqrt_alpha
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(1) if x_0.dim() > 1 else sqrt_one_minus_alpha
        
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise
    
    @torch.no_grad()
    def p_sample(self, x_t, t, aux, clip_denoised=True):
        """
        Reverse diffusion step: denoise sample from timestep t to t-1.
        
        Args:
            x_t: noisy sample at timestep t [batch_size, input_dim]
            t: timestep index (scalar or [batch_size])
            aux: auxiliary information [batch_size, aux_dim]
            clip_denoised: whether to clip denoised values to [-1, 1]
        
        Returns:
            denoised sample at t-1
        """
        # Handle timestep indexing
        if isinstance(t, int):
            batch_size = x_t.shape[0]
            t = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)
        
        # Predict noise
        noise_pred = self.forward(x_t, t, aux)
        
        # Get alpha values
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        alpha_cumprod_prev = self.alphas_cumprod_prev[t]
        
        # Reshape for broadcasting
        if alpha.dim() == 1 and x_t.dim() > 1:
            alpha = alpha.unsqueeze(1)
            alpha_cumprod = alpha_cumprod.unsqueeze(1)
            alpha_cumprod_prev = alpha_cumprod_prev.unsqueeze(1)
        
        # Compute mean
        sqrt_recip_alpha = torch.sqrt(1.0 / alpha)
        model_mean = sqrt_recip_alpha * (x_t - (1.0 - alpha) / torch.sqrt(1.0 - alpha_cumprod) * noise_pred)
        
        # Compute variance
        posterior_var = self.posterior_variance[t]
        if posterior_var.dim() == 1 and x_t.dim() > 1:
            posterior_var = posterior_var.unsqueeze(1)
        
        # Add noise if not at the last step
        if (t > 0).any():
            noise = torch.randn_like(x_t)
            x_t_minus_1 = model_mean + torch.sqrt(posterior_var) * noise
        else:
            x_t_minus_1 = model_mean
        
        if clip_denoised:
            x_t_minus_1 = torch.clamp(x_t_minus_1, -1.0, 1.0)
        
        return x_t_minus_1
    
    @torch.no_grad()
    def sample(self, aux, num_samples=None, clip_denoised=True):
        """
        Generate samples using the reverse diffusion process.
        
        Args:
            aux: auxiliary information (class labels) [num_classes, aux_dim] or single
            num_samples: number of samples per class (if provided)
            clip_denoised: whether to clip denoised values to [-1, 1]
        
        Returns:
            generated samples [num_samples, input_dim]
        """
        device = next(self.parameters()).device
        
        # Handle aux shape
        if aux.dim() == 1:
            aux = aux.unsqueeze(0)
        
        if num_samples is not None:
            # Repeat aux for each sample
            aux = aux.repeat_interleave(num_samples, dim=0)
        
        batch_size = aux.shape[0]
        
        # Start from pure noise
        x_t = torch.randn(batch_size, self.input_dim, device=device)
        
        # Iteratively denoise
        for t in reversed(range(self.n_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x_t = self.p_sample(x_t, t_batch, aux, clip_denoised=clip_denoised)
        
        return x_t
    
    def generate(self, x, aux):
        """
        Generate samples using the full reverse diffusion process.
        This method provides the same interface as DenoiseNet for compatibility
        with DiffusionDataset.
        
        Args:
            x: starting noise [batch_size, input_dim] (ignored, used as reference for shape)
            aux: auxiliary information [batch_size, aux_dim]
        
        Returns:
            generated samples [batch_size, input_dim]
        """
        # Use the batch size from x to determine how many samples to generate
        batch_size = x.shape[0] if x.dim() > 0 else 1
        
        # Extract auxiliary info if needed
        if aux.dim() == 1:
            aux = aux.unsqueeze(0)
        
        # Generate full diffusion samples
        return self.sample(aux, num_samples=None, clip_denoised=True)

