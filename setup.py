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
import numpy as np

def setup():
    parser = argparse.ArgumentParser()

    # Arguments for training
    parser.add_argument("data", type=str, help="Path to the data.")
    parser.add_argument("--dataset", type=str, default="ESC-50", choices=["ESC-50", "FSC22", "UrbanSound8k", "TAU2019", "GTZAN", "ARCA23K-FSD"], help="Dataset to train on.")
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
    parser.add_argument(
        "--synth_data_size",
        type=int,
        default=100,
        help="Number of data samples to generate from the diffusion model per class.",
    )
    parser.add_argument(
        "--cls_epoch",
        type=int,
        default=20,
        help="Number of epochs to train the classifier for.",
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

    elif args.dataset == "UrbanSound8k":
        # args.train_classes = [0, 1, 2, 4, 5, 7, 8]
        args.val_classes = [3, 6, 9]
        args.test_classes = [3, 6, 9]

    elif args.dataset == "TAU2019":
        # args.train_classes = [2, 3, 4, 5, 7, 8, 9]
        args.val_classes = [0, 1, 6]
        args.test_classes = [0, 1, 6]

    elif args.dataset == "GTZAN":
        # args.train_classes = [0, 1, 2, 6, 7, 8, 9]
        args.val_classes = [3, 4, 5]
        args.test_classes = [3, 4, 5]

    elif args.dataset == "ARCA23K-FSD":
        # args.test_classes = ['Female_singing', 'Wind_chime', 'Dishes_and_pots_and_pans', 'Scratching_(performance_technique)', 'Crying_and_sobbing', 'Waves_and_surf', 'Screaming', 'Bark', 'Camera', 'Organ']
        args.test_classes = np.linspace(60, 69, 10)
        args.val_classes = np.linspace(60, 69, 10)
        if args.split == "fold0":
            # args.val_classes = ['Crash_cymbal', 'Run', 'Zipper_(clothing)', 'Acoustic_guitar', 'Gong', 'Knock', 'Train', 'Crack', 'Cough', 'Cricket']
            args.val_classes = np.linspace(0, 9, 10)
        elif args.split == "fold1":
            # args.val_classes = ['Electric_guitar', 'Chewing_and_mastication', 'Keys_jangling', 'Female_speech_and_woman_speaking', 'Crumpling_and_crinkling', 'Skateboard', 'Computer_keyboard', 'Bass_guitar', 'Stream', 'Toilet_flush']
            args.val_classes = np.linspace(10, 19, 10)
        elif args.split == "fold2":
            # args.val_classes = ['Tap', 'Water_tap_and_faucet', 'Squeak', 'Snare_drum', 'Finger_snapping', 'Walk_and_footsteps', 'Meow', 'Rattle_(instrument)', 'Bowed_string_instrument', 'Sawing']
            args.val_classes = np.linspace(20, 29, 10)
        elif args.split == "fold3":
            # args.val_classes = ['Rattle', 'Slam', 'Whoosh_and_swoosh_and_swish', 'Hammer', 'Fart', 'Harp', 'Coin_(dropping)', 'Printer', 'Boom', 'Giggle']
            args.val_classes = np.linspace(30, 39, 10)
        elif args.split == "fold4":
            # args.val_classes = ['Clapping', 'Crushing', 'Livestock_and_farm_animals_and_working_animals', 'Scissors', 'Writing', 'Wind', 'Crackle', 'Tearing', 'Piano', 'Microwave_oven']
            args.val_classes = np.linspace(40, 49, 10)
        elif args.split == "fold5":
            # args.val_classes = ['Trumpet', 'Wind_instrument_and_woodwind_instrument', 'Child_speech_and_kid_speaking', 'Drill', 'Thump_and_thud', 'Drawer_open_or_close', 'Male_speech_and_man_speaking', 'Gunshot_and_gunfire', 'Burping_and_eructation', 'Splash_and_splatter']
            args.val_classes = np.linspace(50, 59, 10)
    return args
