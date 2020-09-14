import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxiliaryNetwork(nn.Module):
    def __init__(self, input_dim, n_labels, dropout):
        super(AuxiliaryNetwork, self).__init__()

        self.input_dim = input_dim  # (seq_len, batch_size, num_directions * hidden_size) [3, 50, 20]
        self.hidden_size = h
        self.n_labels = n_labels
        self.drop_p = dropout

        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, self.n_labels)

        self.dropout1 = torch.nn.Dropout(p=self.drop_p)
        self.dropout2 = torch.nn.Dropout(p=self.drop_p)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        y = F.log_softmax(x)
        return y

