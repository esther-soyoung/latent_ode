import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxiliaryNetwork(nn.Module):
    def __init__(self, input_dim, latent_dim, n_intg, train=True):
        super(AuxiliaryNetwork, self).__init__()

        self.input_dim = input_dim  # [3, 50, 20]
        self.latent_dim = latent_dim  # [3, 50, 10]
        self.n_labels = n_intg  # [3, 50, n_intg]

        self.fc1 = torch.nn.Linear(self.input_dim, self.latent_dim)
        self.fc2 = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.fc3 = torch.nn.Linear(self.latent_dim, self.n_labels)
        self.dropout = torch.nn.Dropout(p=0.25)

    def forward(self, x, backwards=False):
        # x = x.view(-1, self.input_dim)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        y = F.log_softmax(x)
        return y  # [3, 50, n_intg]
