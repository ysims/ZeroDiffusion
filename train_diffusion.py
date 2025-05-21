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

from models.diffusion import Diffusion
from models.classifier import Compatibility
from dataloader.diffusion import DiffusionDataset
import torch

def train_diffusion(config, fixed_config):
    diffusion = Diffusion(
        fixed_config["feature_dim"],
        config["diffusion_hidden_dim"],
        fixed_config["auxiliary_dim"],
    ).to(fixed_config["device"])

    diffusion_optimiser = torch.optim.Adam(
        diffusion.parameters(), lr=config["diffusion_lr"], weight_decay=1e-4
    )

    # Set up dataset loaders
    val_loader = torch.utils.data.DataLoader(
        fixed_config["val_set"], batch_size=config["diffusion_batch_size"], shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        fixed_config["train_set"],
        batch_size=config["diffusion_batch_size"],
        shuffle=True,
    )

    # Main training loop
    for epoch in range(config["diffusion_epoch"]):
        # Batch loop
        train_loss = 0.0
        for _, features, auxiliary in train_loader:
            diffusion_optimiser.zero_grad()

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
        for _, features, auxiliary in val_loader:
            generated = diffusion(
                diffusion.distort(features, epoch / config["diffusion_epoch"]),
                auxiliary,
            )
            loss = torch.nn.functional.mse_loss(generated, features)

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch+1} \t Loss: {train_loss:.6f} \tVal Loss: {loss.item():.6f}"
            )

    return diffusion


def train(config, fixed_config):
    # Train the diffusion model
    diffusion = train_diffusion(config, fixed_config)
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

    # Generate a dataset for the unseen classes using the diffusion network
    gen_set = DiffusionDataset(
        diffusion, fixed_config["feature_dim"], fixed_config["val_auxiliary"]
    )

    train_gen_loader = torch.utils.data.DataLoader(
        gen_set, batch_size=config["classifier_batch_size"], shuffle=True
    )

    train_real_loader = torch.utils.data.DataLoader(
        fixed_config["train_set"],
        batch_size=config["classifier_batch_size"],
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        fixed_config["val_set"],
        batch_size=config["classifier_batch_size"],
        shuffle=True,
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
            optimiser.zero_grad()
            predicted = classifier(data, all_aux).squeeze(1)

            # get the index of labels in all_aux
            labels_ = torch.tensor(
                [
                    torch.where(torch.all(all_aux == label, dim=1))[0][0]
                    for label in labels.detach()
                ]
            ).to(fixed_config["device"])

            loss = criterion(predicted, labels_)
            loss.backward(retain_graph=True)

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
            optimiser.zero_grad()
            predicted = classifier(data, all_aux).squeeze(1)

            # get the index of labels in all_aux
            labels_ = torch.tensor(
                [
                    torch.where(torch.all(all_aux == label, dim=1))[0][0]
                    for label in labels.detach()
                ]
            ).to(fixed_config["device"])

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

        val_aux = fixed_config["val_auxiliary"].float().to(fixed_config["device"])

        classifier.eval()
        for index, data, labels in val_loader:
            # get the index of labels in all_aux
            labels_ = torch.tensor(
                [
                    torch.where(torch.all(val_aux == label, dim=1))[0][0]
                    for label in labels.detach()
                ]
            ).to(fixed_config["device"])

            predicted = classifier(data.to(fixed_config["device"]), val_aux)

            loss = criterion(predicted, labels_)
            val_loss = loss.item() * data.size(0)

            # Get the predicted labels
            _, predicted_labels = torch.max(predicted, dim=1)

            # Calculate the number of correct predictions
            correct_predictions = (predicted_labels == labels_).sum().item()

            # Calculate the accuracy
            accuracy = correct_predictions / labels_.size(0)

            val_acc += accuracy
            val_count += 1
        val_acc /= val_count

        print(
            f"Epoch {epoch+1} \t Train Loss: {(train_loss / (len(train_gen_loader) + len(train_real_loader))):.6f} \t Train acc: {train_acc:.6f} \t Val Loss: {(val_loss / len(val_loader)):.6f} \t Val acc {val_acc:.6f}"
        )

    return {"mean_accuracy": val_acc}
