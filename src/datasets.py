import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import datetime
from itertools import product
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from .utils import *

class ConditionedDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, conditions=None, conditioner=None, num_samples=1):
        self.inputs = inputs
        self.conditions = conditions ##raw
        self.conditioner = conditioner
        self.num_samples = num_samples

    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        input_ = self.inputs[idx]
        condition_ = self.conditioner.transform({key: condition[[idx]] for key, condition in self.conditions.items()}, num_samples=self.num_samples).squeeze()
        return torch.tensor(input_).float(), torch.tensor(condition_).float()
    
class ContexedDataset(torch.utils.data.Dataset):
    def __init__(self, targets, inputs, contexts, conditioner=None):
        self.targets = targets
        self.inputs = inputs
        self.contexts = contexts

        self.conditioner = conditioner

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        input_ = self.inputs[idx]
        context_ = self.conditioner.transform({key: context[[idx]] for key, context in self.contexts.items()}).squeeze()
        target_ = self.targets[idx]
        return torch.tensor(input_).float(), torch.tensor(context_).float(), torch.tensor(target_).float()