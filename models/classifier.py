# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# classifier.py
#
# - Attribute Label Embedding (ALE) compatibility function
# - Normalized Zero-Shot evaluation
#
# Written by cetinsamet -*- cetin.samet@metu.edu.tr
# December, 2019
# --------------------------------------------------
import torch
import torch.nn.functional as F
import torch.nn as nn


def init_layer(layer):
    torch.nn.init.xavier_normal_(layer.weight, gain=0.1)

    if layer.bias is not None:
        layer.bias.data.fill_(0.0)


class Compatibility(nn.Module):
    def __init__(self, d_in, d_out):
        super(Compatibility, self).__init__()

        hidden_units = d_in
        self.fc1 = nn.Linear(d_in, hidden_units, bias=False)
        self.fc2 = nn.Linear(hidden_units, d_out, bias=False)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, x1, x2):
        """
        :param x1: (num_samples, num_segments, d_in)
        :param x2: (num_classes, d_out)
        :return: (num_samples, num_classes)
        """

        # x1 = torch.mean(x1, dim=1, keepdim=False)  # (num_samples, d_in)

        x1 = torch.tanh(self.fc1(x1))
        x1 = self.fc2(x1)  # (num_samples, d_out)

        x2 = x2.transpose(0, 1)  # (d_out, num_classes)

        output = x1.matmul(x2)  # (num_samples, num_classes)

        return output


# Determine if this is a correct prediction
# Is correct if predicted is closest to actual out of all_aux,
# using cosine similarity
# predicted and actual are vectors, and all_aux is a list of vectors
def evaluate(model, predicted, actual, all_aux):
    # Find the closest in all_aux to the predicted
    # using cosine similarity
    closest = None
    closest_dist = -1
    for aux in all_aux:
        dist = F.cosine_similarity(predicted, aux)
        if dist > closest_dist:
            closest_dist = dist
            closest = aux

    # If the closest is the actual, then the prediction is correct
    return closest == actual
