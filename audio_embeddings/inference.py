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

# Function to convert a .wav file to an audio embedding using a given model

import torch
import numpy as np
from dataset.dataset_utils import wavfile_to_examples

# Takes a .wav file and returns the semantic audio vector to be used for that sound
# Sound embeddings of each 1-second clip are created, and these are averaged to create the final embedding (Xie, 2021)
def audio_to_embedding(wav_file, model, device, channels):
    with torch.no_grad():
        if model.name == "Inception":
            bins = 480
        else:
            bins = 64
        # Convert the wav file given to the input type for vggish
        wav_embedding = wavfile_to_examples(wav_file, bins)
        if channels == 1:
            input = np.array([[wav_embedding[0]]])
        elif channels == 3:
            input = np.array([[wav_embedding[0], wav_embedding[0], wav_embedding[0]]])
        input = torch.from_numpy(input).float().to(device)
        embedding = model.inference(input)

        return embedding
