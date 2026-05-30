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

# The main training functions for the ZeroDiffusion method 

from models.denoise import DenoiseNet
from models.ddpm import DDPM
from models.classifier import Compatibility
from dataloader.diffusion import DiffusionDataset
import torch
import os


def _build_loader_kwargs(device, shuffle=True):
    use_cuda = isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available()
    workers = min(8, os.cpu_count() or 1)
    kwargs = {
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": use_cuda,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return kwargs


def _label_indices_from_aux(labels, aux_bank):
    # Match each label vector in a batch to its row index in the auxiliary bank.
    matches = torch.isclose(
        labels.unsqueeze(1), aux_bank.unsqueeze(0), atol=1e-6, rtol=1e-6
    ).all(dim=-1)
    if not torch.all(matches.any(dim=1)):
        raise RuntimeError("At least one label vector was not found in auxiliary bank.")
    return matches.float().argmax(dim=1)

def train_diffusion(config, fixed_config):
    diffusion = DenoiseNet(
        fixed_config["feature_dim"],
        config["diffusion_hidden_dim"],
        fixed_config["auxiliary_dim"],
    ).to(fixed_config["device"])

    diffusion_optimiser = torch.optim.Adam(
        diffusion.parameters(), lr=config["diffusion_lr"], weight_decay=1e-4
    )

    loader_kwargs = _build_loader_kwargs(fixed_config["device"], shuffle=True)

    # Set up dataset loaders
    val_loader = torch.utils.data.DataLoader(
        fixed_config["val_set"], batch_size=config["diffusion_batch_size"], **loader_kwargs
    )
    train_loader = torch.utils.data.DataLoader(
        fixed_config["train_set"],
        batch_size=config["diffusion_batch_size"],
        **loader_kwargs,
    )

    # Main training loop
    for epoch in range(config["diffusion_epoch"]):
        diffusion.train()
        # Batch loop
        train_loss = 0.0
        for _, features, auxiliary in train_loader:
            features = features.to(fixed_config["device"], non_blocking=True)
            auxiliary = auxiliary.to(fixed_config["device"], non_blocking=True)
            diffusion_optimiser.zero_grad(set_to_none=True)
            # Forward pass
            generated = diffusion(
                diffusion.distort(features, epoch / config["diffusion_epoch"]),
                auxiliary,
            )

            loss = torch.nn.functional.mse_loss(generated, features)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
            diffusion_optimiser.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Evaluation loop
        diffusion.eval()
        loss = 0.0
        with torch.no_grad():
            for _, features, auxiliary in val_loader:
                features = features.to(fixed_config["device"], non_blocking=True)
                auxiliary = auxiliary.to(fixed_config["device"], non_blocking=True)
                generated = diffusion(
                    diffusion.distort(features, epoch / config["diffusion_epoch"]),
                    auxiliary,
                )
                loss += torch.nn.functional.mse_loss(generated, features).item()

        loss /= len(val_loader)

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch+1} \t Loss: {train_loss:.6f} \tVal Loss: {loss:.6f}"
            )

    return diffusion


def train_ddpm(config, fixed_config):
    """Train a DDPM diffusion model"""
    ddpm = DDPM(
        input_dim=fixed_config["feature_dim"],
        aux_dim=fixed_config["auxiliary_dim"],
        hidden_dim=config["diffusion_hidden_dim"],
        n_layers=config.get("ddpm_n_layers", 4),
        n_timesteps=config.get("ddpm_n_timesteps", 1000),
        dropout=config.get("ddpm_dropout", 0.3),
        use_layernorm=config.get("ddpm_use_layernorm", True),
    ).to(fixed_config["device"])

    ddpm_optimiser = torch.optim.Adam(
        ddpm.parameters(), lr=config["diffusion_lr"], weight_decay=1e-4
    )

    loader_kwargs = _build_loader_kwargs(fixed_config["device"], shuffle=True)

    # Set up dataset loaders
    val_loader = torch.utils.data.DataLoader(
        fixed_config["val_set"], batch_size=config["diffusion_batch_size"], **loader_kwargs
    )
    train_loader = torch.utils.data.DataLoader(
        fixed_config["train_set"],
        batch_size=config["diffusion_batch_size"],
        **loader_kwargs,
    )

    # Main training loop
    for epoch in range(config["diffusion_epoch"]):
        ddpm.train()
        # Batch loop
        train_loss = 0.0
        for _, features, auxiliary in train_loader:
            features = features.to(fixed_config["device"], non_blocking=True).float()
            auxiliary = auxiliary.to(fixed_config["device"], non_blocking=True).float()
            
            ddpm_optimiser.zero_grad(set_to_none=True)
            
            # Sample random timesteps
            batch_size = features.shape[0]
            t = torch.randint(0, config.get("ddpm_n_timesteps", 1000), (batch_size,), 
                             device=fixed_config["device"])
            
            # Forward diffusion: add noise
            x_t, noise = ddpm.q_sample(features, t)
            
            # Predict noise
            noise_pred = ddpm(x_t, t, auxiliary)
            
            # Loss: MSE between predicted and actual noise
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), max_norm=1.0)
            ddpm_optimiser.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Evaluation loop
        ddpm.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _, features, auxiliary in val_loader:
                features = features.to(fixed_config["device"], non_blocking=True).float()
                auxiliary = auxiliary.to(fixed_config["device"], non_blocking=True).float()
                
                # Sample random timesteps
                batch_size = features.shape[0]
                t = torch.randint(0, config.get("ddpm_n_timesteps", 1000), (batch_size,),
                                 device=fixed_config["device"])
                
                # Forward diffusion
                x_t, noise = ddpm.q_sample(features, t)
                
                # Predict noise
                noise_pred = ddpm(x_t, t, auxiliary)
                
                # Loss
                val_loss += torch.nn.functional.mse_loss(noise_pred, noise).item()

        val_loss /= len(val_loader)

        if epoch % 100 == 0:
            print(
                f"DDPM Epoch {epoch+1} \t Loss: {train_loss:.6f} \tVal Loss: {val_loss:.6f}"
            )

    return ddpm


def train(config, fixed_config, method="dae"):
    """
    Train diffusion model and classifier for zero-shot learning.
    
    Args:
        config: Training configuration dict
        fixed_config: Fixed configuration dict
        method: "dae" for DenoiseNet or "ddpm" for DDPM model
    """
    if torch.cuda.is_available() and isinstance(fixed_config["device"], str) and fixed_config["device"].startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    # Train the diffusion model
    if method == "dae":
        print(f"Training DenoiseNet diffusion model...")
        diffusion = train_diffusion(config, fixed_config)
    elif method == "ddpm":
        print(f"Training DDPM diffusion model...")
        diffusion = train_ddpm(config, fixed_config)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'dae' or 'ddpm'")
    
    diffusion.eval()

    classifier = Compatibility(
        fixed_config["feature_dim"],
        fixed_config["auxiliary_dim"],
    ).to(fixed_config["device"])

    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(
        classifier.parameters(),
        lr=config["classifier_learning_rate"],
        weight_decay=1e-5,
    )

    # Get the average norm of the training set
    norm = torch.tensor(
        [
            features.norm() if features.dim() == 1 else torch.norm(features, dim=1).mean()
            for _, features, _ in fixed_config["train_set"]
        ]
    ).mean().to(fixed_config["device"])

    # Generate the dataset for the unseen classes
    gen_set = DiffusionDataset(
        diffusion,
        fixed_config["feature_dim"],
        fixed_config["val_auxiliary"],
        config["classifier_dataset_size"],
        norm,
        generation_batch_size=config.get("generation_batch_size", 128),
    )

    all_aux = (
        torch.cat(
            [
                fixed_config["val_auxiliary"],
                fixed_config["train_auxiliary"],
            ],
            dim=0,
        )
        .float()
        .to(fixed_config["device"])
    )

    classifier_loader_kwargs = _build_loader_kwargs(fixed_config["device"], shuffle=True)

    train_gen_loader = torch.utils.data.DataLoader(
        gen_set,
        batch_size=config["classifier_batch_size"],
        **classifier_loader_kwargs,
    )

    train_real_loader = torch.utils.data.DataLoader(
        fixed_config["train_set"],
        batch_size=config["classifier_batch_size"],
        **classifier_loader_kwargs,
    )

    val_loader = torch.utils.data.DataLoader(
        fixed_config["val_set"],
        batch_size=config["classifier_batch_size"],
        **classifier_loader_kwargs,
    )

    # Keep track of val accuracy for when returning
    val_acc = 0.0

    # Main training loop
    for epoch in range(config["classifier_epoch"]):
        # Batch loop
        train_loss = 0.0
        train_count = 0.0
        train_acc = 0.0

        classifier.train()
        for data, labels in train_gen_loader:
            data = data.to(fixed_config["device"], non_blocking=True)
            labels = labels.to(fixed_config["device"], non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            predicted = classifier(data, all_aux).squeeze(1)

            labels_ = _label_indices_from_aux(labels, all_aux).long()

            loss = criterion(predicted, labels_)
            loss.backward()

            optimiser.step()

            train_loss += loss.item()

            # Get the predicted labels
            _, predicted_labels = torch.max(predicted, dim=1)

            # Calculate the number of correct predictions
            correct_predictions = (predicted_labels == labels_).sum().item()

            # Calculate the accuracy
            accuracy = correct_predictions / labels_.size(0)

            # Add the accuracy to the total accuracy
            train_acc += accuracy
            train_count += 1

        for _, data, labels in train_real_loader:
            data = data.to(fixed_config["device"], non_blocking=True)
            labels = labels.to(fixed_config["device"], non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            predicted = classifier(data, all_aux).squeeze(1)

            labels_ = _label_indices_from_aux(labels, all_aux).long()

            loss = criterion(predicted, labels_)
            loss.backward()

            optimiser.step()

            train_loss += loss.item()

            # Get the predicted labels
            _, predicted_labels = torch.max(predicted, dim=1)

            # Calculate the number of correct predictions
            correct_predictions = (predicted_labels == labels_).sum().item()

            # Calculate the accuracy
            accuracy = correct_predictions / labels_.size(0)

            # Add the accuracy to the total accuracy
            train_acc += accuracy
            train_count += 1

        train_acc /= train_count

        # Evaluation loop
        val_count = 0.0
        val_loss = 0.0
        val_acc = 0.0

        val_aux = fixed_config["val_auxiliary"].float().to(fixed_config["device"])

        classifier.eval()
        with torch.no_grad():
            for index, data, labels in val_loader:
                data = data.to(fixed_config["device"], non_blocking=True)
                labels = labels.to(fixed_config["device"], non_blocking=True)
                labels_ = _label_indices_from_aux(labels, val_aux).long()

                predicted = classifier(data, val_aux)

                loss = criterion(predicted, labels_)
                val_loss += loss.item()

                # Get the predicted labels
                _, predicted_labels = torch.max(predicted, dim=1)

                # Calculate the number of correct predictions
                correct_predictions = (predicted_labels == labels_).sum().item()

                val_acc += correct_predictions
                val_count += labels_.size(0)
        val_acc /= val_count

        print(
            f"Epoch {epoch+1} \t Train Loss: {(train_loss / (len(train_gen_loader) + len(train_real_loader))):.6f} \t Train acc: {train_acc:.6f} \t Val Loss: {(val_loss / len(val_loader)):.6f} \t Val acc {val_acc:.6f}"
        )

    return {"mean_accuracy": val_acc}
