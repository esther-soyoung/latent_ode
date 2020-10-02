import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxiliaryBlock(nn.Module):
    def __init__(self, input_dim, n_intg, train=True):
        super(AuxiliaryBlock, self).__init__()

        self.input_dim = input_dim  # 20
        self.latent_dim = 10
        self.latent_dim2 = 5
        self.n_labels = n_intg  # [3, 50, n_intg]

        self.fc1 = torch.nn.Linear(self.input_dim, self.latent_dim)
        self.fc2 = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.fc3 = torch.nn.Linear(self.latent_dim, self.latent_dim2)
        self.fc4 = torch.nn.Linear(self.latent_dim2, self.latent_dim2)
        self.fc5 = torch.nn.Linear(self.latent_dim2, self.n_labels)
        self.dropout = torch.nn.Dropout(p=0.25)

    def forward(self, x, backwards=False):
        # x = x.view(-1, self.input_dim)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        # x = self.fc4(x)
        # x = F.relu(x)
        x = self.dropout(x)
        out = self.fc5(x)

        return out  # [3, 50, n_intg]
