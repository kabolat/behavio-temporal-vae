import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import datetime
from itertools import product
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from .utils import *

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

class Conditioner():
    def __init__(self, tags, supports, types, condition_set=None):
        self.tags = tags
        self.supports = supports
        self.types = types
        self.cond_dim = 0
        self.transformers = {}
        self.init_transformers(condition_set)

    def add_transformer(self, tag, support, typ, data=None):
        if typ == "circ":
            self.transformers[tag] = CircularTransformer(max_conds=np.max(support), min_conds=np.min(support))
            self.cond_dim += 2
        elif typ == "cat":
            self.transformers[tag] = OneHotEncoder(sparse_output=False).fit(data)
            self.cond_dim += self.transformers[tag].categories_[0].shape[0]
        elif typ == "cont":
            self.transformers[tag] = MinMaxTransformer(feature_range=(-1, 1)).fit(data)
            self.cond_dim += 1
        elif typ == "ord":
            ## always give the ascending support!
            self.transformers[tag] = OrdinalEncoder(categories=[support]).fit(data)
            self.cond_dim += 1
        elif typ == "dir":
            num_dims = data.shape[1]
            self.transformers[tag] = DirichletTransformer(num_dims=num_dims, transform_style="sample")
            self.cond_dim += num_dims
        else:
            raise ValueError("Unknown type.")
    
    def init_transformers(self, data):
        for tag, support, typ in zip(self.tags, self.supports, self.types):
            self.add_transformer(tag, support, typ, data[tag])
    
    def add_condition(self, tag, support, typ, data=None):
        self.tags.append(tag)
        self.supports.append(support)
        self.types.append(typ)
        self.add_transformer(tag, support, typ, data)
    
    def transform(self, data):
        transformed_data = []
        for tag in self.tags: 
            data_ = data[tag]
            transformed_data.append(self.transformers[tag].transform(data_))
        return np.concatenate(transformed_data, axis=1)
    
    def get_random_conditions(self, num_samples=1, random_seed=None):
        if random_seed is not None: np.random.seed(random_seed)
        random_conditions = {}
        for tag, typ, support in zip(self.tags, self.types, self.supports):
            if typ == "circ":
                random_conditions[tag] = np.random.randint(self.transformers[tag].min_conds, self.transformers[tag].max_conds+1, num_samples)[...,None]
            elif typ == "cat" or typ == "ord":
                random_conditions[tag] = np.random.choice(self.transformers[tag].categories_[0], num_samples)[...,None]
            elif typ == "cont":
                random_conditions[tag] = self.transformers[tag].inverse_transform(np.random.rand(num_samples)[:, np.newaxis]).squeeze(-1)[...,None]
            elif typ == "dir":
                rnd_doc_length = np.random.uniform(low=support[0], high=support[1], size=num_samples)
                random_conditions[tag] = np.random.dirichlet(alpha=[1.0]*self.transformers[tag].num_dims, size=num_samples)*rnd_doc_length[:, np.newaxis]
            else:
                raise ValueError("Unknown type.")
        condition_set = self.transform(random_conditions)
        return condition_set, random_conditions

class ConditionedDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, conditions=None, conditioner=None):
        self.inputs = inputs
        self.conditions = conditions ##raw
        self.conditioner = conditioner

    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        input_ = self.inputs[idx]
        condition_ = self.conditioner.transform({key: conditon[[idx]] for key, conditon in self.conditions.items()}).squeeze()
        return torch.tensor(input_).float(), torch.tensor(condition_).float()