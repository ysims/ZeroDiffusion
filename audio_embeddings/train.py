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

# This script is used to train a selected model on the given dataset to produce audio embeddings for zero-shot learning.

from models.VGGish import VGGNet
from models.YAMNet import YAMNet
from models.Inception import InceptionV4
from dataset.dataset import ZSLESC50Dataset, ZSLFSC22Dataset
from EarlyStopping import EarlyStopping
from setup import setup

import os
import torch
import torch.nn as nn

args = setup()

# Make folder if args.save_path folder does not exist
# Remove file name from args.save_path
if not os.path.exists(os.path.dirname(args.save_path)):
    os.makedirs(os.path.dirname(args.save_path))

if args.model == "VGGish":
    model = VGGNet(num_classes=len(args.train_classes)).to(args.device)
    channels = 1
    num_mel_bins = 64
elif args.model == "YAMNet":
    model = YAMNet(num_classes=len(args.train_classes), channels=1).to(args.device)
    channels = 1
    num_mel_bins = 64
elif args.model == "YAMNet3":
    model = YAMNet(num_classes=len(args.train_classes), channels=3).to(args.device)
    channels = 3
    num_mel_bins = 64
elif args.model == "Inception":
    model = InceptionV4(num_classes=len(args.train_classes)).to(args.device)
    channels = 3
    num_mel_bins = 480

# pos_weight = torch.ones([40]).float().to(args.device)  # All weights are equal to 1
# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

if args.dataset == "ESC-50":
    # Create the ESC-50 dataset
    dataset = ZSLESC50Dataset(
        args.device, args.data_path, args.train_classes, channels, num_mel_bins
    )
elif args.dataset == "FSC22":
    # Create the FSC22 dataset
    dataset = ZSLFSC22Dataset(
        args.train_classes, args.device, channels, num_mel_bins, args.data_path
    )
else:
    raise ValueError("Dataset not supported.")

# Get the size of each dataset
val_size = int(len(dataset) * 0.2)
train_size = len(dataset) - val_size

# Split the dataset into train and validation
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=args.batch_size, shuffle=True
)

early_stopping = EarlyStopping(patience=args.patience)

best_val_acc = 0

# Main training loop
for e in range(args.epoch):

    # Batch loop
    train_loss = 0.0
    train_count = 0.0
    train_acc = 0.0
    model.train()
    for data, labels in train_loader:
        optimiser.zero_grad()

        target = model(data)
        loss = criterion(target, labels)
        loss.backward()

        optimiser.step()

        train_loss += loss.item()

        target = torch.sigmoid(target)
        for i in range(len(labels)):
            train_acc += labels[i] == torch.argmax(target[i])
            train_count += 1
    train_acc /= train_count

    # Evaluation loop
    acc = 0.0
    count = 0.0
    val_loss = 0.0
    model.eval()
    for data, labels in val_loader:
        target = model(data)
        loss = criterion(target, labels)
        val_loss = loss.item() * data.size(0)
        target = torch.sigmoid(target)
        for i in range(len(labels)):
            acc += labels[i] == torch.argmax(target[i])
            count += 1
    acc /= count

    print(
        f"Epoch {e+1} \t Train Loss: {(train_loss / len(train_loader)):.6f} \t Train acc: {train_acc:.6f} \t Val Loss: {(val_loss / len(val_loader)):.6f} \t Val acc {acc:.6f}"
    )

    if acc > best_val_acc:
        best_val_acc = acc
        torch.save(model.state_dict(), args.save_path)

    # Update early stopping
    if early_stopping.stop(val_loss):
        print(f"Early stopping at epoch {e+1}. Best model has accuracy {best_val_acc}.")
        break
