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

# This file contains two dataset classes (ESC-50 and FSC22). They use pickle files from data/load_esc50.py and data/load_fsc22.py.

import os
import re
import numpy as np

from dataset.dataset_utils import wavfile_to_examples

import torch
from torch.utils.data import Dataset


class ZSLESC50Dataset(Dataset):
    # data_path: to a pickle file with
    #   data["audio"]
    #   data["labels"]
    def __init__(self, device, data_path, train_classes, channels, bins):
        self.audio = []
        self.labels = []
        self.device = device

        for file in os.listdir(data_path):
            # Split the file name up so the data is known
            info = re.split("-|\.", file)
            # fold = info[0]         # random folds are defined to group data randomly
            # clip_id = info[1]      # id of the clip
            # take = info[2]         # some clips are separated into A,B,C
            target = int(info[3])  # class, 0-49
            if train_classes.count(target) > 0:
                wav_data = wavfile_to_examples(os.path.join(data_path, file), bins)
                if channels == 3:
                    self.audio.append(np.array([wav_data[0], wav_data[0], wav_data[0]]))
                elif channels == 1:
                    self.audio.append(wav_data)
                self.labels.append(
                    train_classes.index(target)
                )  # only have numbers in the range of [0, len(train_classes)]
        self.length = len(self.audio)

        self.audio = np.array(self.audio)
        self.labels = np.array(self.labels)

        self.audio = torch.from_numpy(self.audio).float().to(self.device)
        self.labels = torch.from_numpy(self.labels).long().to(self.device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (self.audio[index], self.labels[index])

class ZSLFSC22Dataset(Dataset):
    # data_path: to a pickle file with
    #   data["audio"]
    #   data["labels"]
    def __init__(self, train_classes, device="cuda", channels=1, bins=64, data_path="../data/FSC22/audio"):
        self.audio = []
        self.labels = []
        self.device = device

        for file in os.listdir(data_path):
            # Get the class from the file name
            target = int(file.split("_")[0])
            
            # Check if we want to train with this class
            if train_classes.count(target) > 0:
                wav_data = wavfile_to_examples(os.path.join(data_path, file), bins)
                if channels == 3:
                    self.audio.append(np.array([wav_data[0], wav_data[0], wav_data[0]]))
                elif channels == 1:
                    self.audio.append(wav_data)
                self.labels.append(
                    train_classes.index(target)
                )  # only have numbers in the range of [0, len(train_classes)]
        self.length = len(self.audio)

        self.audio = np.array(self.audio)
        self.labels = np.array(self.labels)

        self.audio = torch.from_numpy(self.audio).float().to(self.device)
        self.labels = torch.from_numpy(self.labels).long().to(self.device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (self.audio[index], self.labels[index])
