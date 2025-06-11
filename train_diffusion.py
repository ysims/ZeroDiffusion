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
from models.warp import WARP

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def compute_mmd(x, y, sigma=1.0):
    def rbf(x1, x2):
        diff = x1.unsqueeze(1) - x2.unsqueeze(0)
        dist_sq = (diff ** 2).sum(2)
        return torch.exp(-dist_sq / (2 * sigma ** 2))

    k_xx = rbf(x, x)
    k_yy = rbf(y, y)
    k_xy = rbf(x, y)

    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

def variance_loss(gen, real):
    var_gen = gen.var(dim=0)
    var_real = real.var(dim=0)
    return torch.nn.functional.mse_loss(var_gen, var_real)

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

            loss = 1.0 * torch.nn.functional.mse_loss(generated, features)
            loss += 1.0 * compute_mmd(generated, features)
            loss += 0.1 * variance_loss(generated, features)
            loss += 0.2 * torch.nn.functional.mse_loss(generated.mean(dim=0), features.mean(dim=0))

            cos_sim = torch.nn.functional.cosine_similarity(generated, features, dim=1)
            cos_loss = 1.0 - cos_sim.mean()
            loss += 2.0 * cos_loss

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)

            diffusion_optimiser.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Evaluation loop
        diffusion.eval()
        loss = 0.0
        for _, features, auxiliary in val_loader:
            generated = diffusion(
                diffusion.distort(features, epoch / config["diffusion_epoch"]),
                auxiliary,
            )
            loss += torch.nn.functional.mse_loss(generated, features)

        loss /= len(val_loader)

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch+1} \t Loss: {train_loss:.6f} \tVal Loss: {loss:.6f}"
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

    # criterion = WARP
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(
        classifier.parameters(),
        lr=config["classifier_learning_rate"],
        weight_decay=1e-5,
    )

    # Get the average norm of the training set
    # Get the average norm of the training set
    norm = torch.tensor(
        [
            features.norm() if features.dim() == 1 else torch.norm(features, dim=1).mean()
            for _, features, _ in fixed_config["train_set"]
        ]
    ).mean().to(fixed_config["device"])

    # Generate the dataset for the unseen classes
    gen_set = DiffusionDataset(
        diffusion, fixed_config["feature_dim"], fixed_config["val_auxiliary"], config["classifier_dataset_size"], norm
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
    # Collect data and labels from the DataLoader
    # gen_data_list = []
    # gen_labels_list = []

    # for data, labels in train_gen_loader:
    #     labels_ = torch.tensor(
    #         [
    #             torch.where(torch.all(all_aux == label, dim=1))[0][0]
    #             for label in labels.detach()
    #         ]
    #     ).to(fixed_config["device"])
    #     # Collapse the batch dimension
    #     data = data.view(data.size(0), -1)  # Flatten all dimensions except the batch size
    #     labels_ = labels_.view(-1)  # Flatten labels if needed
    #     gen_data_list.append(data.cpu().detach().numpy())
    #     gen_labels_list.append(labels_.cpu().detach().numpy())

    # real_data_list = []
    # real_labels_list = []
    # # Compare with real unseen data
    # for _, data, labels in val_loader:
    #     labels_ = torch.tensor(
    #         [
    #             torch.where(torch.all(all_aux == label, dim=1))[0][0]
    #             for label in labels.detach()
    #         ]
    #     ).to(fixed_config["device"])
    #     # Collapse the batch dimension
    #     data = data.view(data.size(0), -1)
    #     labels_ = labels_.view(-1)
    #     real_data_list.append(data.cpu().detach().numpy())
    #     real_labels_list.append(labels_.cpu().detach().numpy())


    # # Convert lists to numpy arrays
    # gen_data = np.concatenate(gen_data_list, axis=0)
    # gen_labels = np.concatenate(gen_labels_list, axis=0)
    # real_data = np.concatenate(real_data_list, axis=0)
    # real_labels = np.concatenate(real_labels_list, axis=0)

    # # Perform t-SNE visualization

    # tsne = TSNE(n_components=2, perplexity=10, n_iter=1000)
    # combined_data = np.concatenate((gen_data, real_data), axis=0)
    # gen_data_embedded = tsne.fit_transform(combined_data)

    # combined_labels = np.concatenate((gen_labels, real_labels), axis=0)
    # plt.figure(figsize=(10, 10))
    # sns.scatterplot(
    #     x=gen_data_embedded[:, 0],
    #     y=gen_data_embedded[:, 1],
    #     hue=combined_labels,
    #     palette="Set1",
    #     legend="full",
    # )
    # plt.title("t-SNE of Generated Data")
    # plt.show()

    # Calculate the mean distance between generated and real data
    # gen_data_tensor = torch.tensor(gen_data, dtype=torch.float32).to(fixed_config["device"])
    # real_data_tensor = torch.tensor(real_data, dtype=torch.float32).to(fixed_config["device"])
    # # make same size
    # if gen_data_tensor.size(0) != real_data_tensor.size(0):
    #     min_size = min(gen_data_tensor.size(0), real_data_tensor.size(0))
    #     gen_data_tensor = gen_data_tensor[:min_size]
    #     real_data_tensor = real_data_tensor[:min_size]

    # mean_distance = torch.mean(torch.norm(gen_data_tensor - real_data_tensor, dim=1))
    # # Calculate the covariance difference
    # gen_cov = torch.cov(gen_data_tensor.T)
    # real_cov = torch.cov(real_data_tensor.T)
    # cov_diff = torch.norm(gen_cov - real_cov)
    # # Calculate cosine similarity
    # gen_data_tensor = torch.nn.functional.normalize(gen_data_tensor, dim=1)
    # real_data_tensor = torch.nn.functional.normalize(real_data_tensor, dim=1)
    # cosine_similarity = torch.nn.functional.cosine_similarity(gen_data_tensor, real_data_tensor, dim=1)
    # mean_cosine_similarity = torch.mean(cosine_similarity)

    # print("Mean cosine similarity:", mean_cosine_similarity.item())
    # print("Mean distance:", mean_distance.item())
    # print("Covariance diff:", cov_diff.item())

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
        val_acc = 0.0

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
