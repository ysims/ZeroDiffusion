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

# The setup parameters for training the zero-shot learning diffusion method.

import argparse
import torch

def setup():
    parser = argparse.ArgumentParser()

    # Arguments for training
    parser.add_argument("data", type=str, help="Path to the data.")
    parser.add_argument("--dataset", type=str, default="ESC-50", help="Dataset we are running.")
    parser.add_argument(
        "--split",
        type=str,
        default="fold0",
        help="foldx where x is the val fold, or test for running the test fold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on. Auto will check if cuda can be used, else it will use cpu.",
    )

    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hold out 4th fold, do 4-fold cross-validation on the remaining folds
    if args.dataset == "ESC-50":
        args.val_classes = []
        if args.split == "fold0":
            args.val_classes = [27, 46, 38, 3, 29, 48, 40, 31, 2, 35]
        elif args.split == "fold1":
            args.val_classes = [22, 13, 39, 49, 32, 26, 42, 21, 19, 36]
        elif args.split == "fold2":
            args.val_classes = [23, 41, 14, 24, 33, 30, 4, 17, 10, 45]
        elif args.split == "fold3":
            args.val_classes = [47, 34, 20, 44, 25, 6, 7, 1, 28, 18]
        args.test_classes = [43, 5, 37, 12, 9, 0, 11, 8, 15, 16]
    elif args.dataset == "FSC22":
        args.val_classes = []
        args.test_classes = [5, 7, 15, 17, 21, 23, 26]
        if args.split != "test":
            args.val_classes = [6, 8, 9, 12, 13, 18, 22]

    return args
