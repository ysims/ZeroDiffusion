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

# This script is used to visualise the audio embedding space for an unseen class set

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
import argparse

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset",
    type=str,
    help="Which dataset to use.",
)
parser.add_argument(
    "split",
    type=str,
    help="Which split to use.",
)
parser.add_argument(
    "--file",
    type=str,
    default=None,
    help="Pickle file to load.",
)
args = parser.parse_args()


# Open the pickle and get the embeddings
with open(args.file, "rb") as f:
    data = pickle.load(f)

# Get val classes
if args.dataset == "ESC-50":
    if args.split == "Fold 0":
        classes = [27, 46, 38, 3, 29, 48, 40, 31, 2, 35]
    elif args.split == "Fold 1":
        classes = [22, 13, 39, 49, 32, 26, 42, 21, 19, 36]
    elif args.split == "Fold 2":
        classes = [23, 41, 14, 24, 33, 30, 4, 17, 10, 45]
    elif args.split == "Fold 3":
        classes = [47, 34, 20, 44, 25, 6, 7, 1, 28, 18]
    else:
        classes = [43, 5, 37, 12, 9, 0, 11, 8, 15, 16]
elif args.dataset == "FSC22":
    classes = [5, 7, 15, 17, 21, 23, 26]
    if args.split != "Test":
        classes = [6, 8, 9, 12, 13, 18, 22]

# Get the embeddings
all_labels = np.array(data["labels"])
all_features = np.array([list(d.to("cpu")[0]) for d in data["features"]])
all_auxiliary = np.array(data["auxiliary"])

# Get only the embeddings for the val classes
val_indices = np.where(np.isin(all_labels, classes))[0]
all_labels = all_labels[val_indices]
all_features = all_features[val_indices]

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=0)
all_features = tsne.fit_transform(all_features)

# Plot the embeddings
plt.figure(figsize=(10, 10))
colors = plt.cm.tab20.colors
for i, class_ in enumerate(classes):
    indices = np.where(all_labels == class_)[0]
    plt.scatter(all_features[indices, 0], all_features[indices, 1], label=class_, color=colors[i])

# Add legend
custom_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(len(classes))]
plt.legend(custom_lines, classes, title="Classes", fontsize=22, title_fontsize=22, bbox_to_anchor=(1, 1), loc="upper left")

# Fit so the legend is not cut off
plt.tight_layout()

# Set aspect ratio
x_range = all_features[:, 0].max() - all_features[:, 0].min()
y_range = all_features[:, 1].max() - all_features[:, 1].min()
aspect_ratio = x_range / y_range
plt.gca().set_aspect(aspect_ratio)

# Save the plot
plt.savefig(f"{args.dataset}-{args.split}-tSNE.png")

# Show the plot
plt.show()
