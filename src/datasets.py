import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import datetime
from itertools import product
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from .utils import *

class ConditionedDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, conditions=None, conditioner=None):
        self.inputs = inputs
        self.conditions = conditions ##raw
        self.conditioner = conditioner

    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        input_ = self.inputs[idx]
        condition_ = self.conditioner.transform({key: condition[[idx]] for key, condition in self.conditions.items()}).squeeze()
        return torch.tensor(input_).float(), torch.tensor(condition_).float()