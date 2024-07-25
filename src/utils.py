import torch
from torch.nn.functional import softplus
from torch import inf
import torch
from pandas import Timestamp
from datetime import timedelta
import numpy as np
import os, json

def to_positive(xtilde): return torch.nn.functional.softplus(xtilde, beta=1, threshold=5)

def from_positive(x): return x + torch.log(-torch.expm1(-x))

def to_sigma(sigmatilde): return to_positive(sigmatilde)

def from_sigma(sigma): return from_positive(sigma)

def matrix_normalizer(matrix):
    return matrix / torch.linalg.norm(matrix, dim=0, ord=2, keepdim=True)

def log_prob(dist="Normal", params=None, targets=None):
    if dist=="Normal":
        return -0.5*torch.log(2*torch.tensor(torch.pi)) - torch.log(params["sigma"]) - 0.5*((targets-params["mu"])/params["sigma"])**2
    elif dist=="MultivariateNormal":
        ...
    else:
        raise ValueError("Unknown distribution.")

def kl_divergence(dist="Normal", params=None, prior_params=None):
    if dist=="Normal":
        var_ratio = (params["sigma"] / prior_params["sigma"]).pow(2)
        t1 = ((params["mu"] - prior_params["mu"]) / prior_params["sigma"]).pow(2)
        return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
    else:
        raise ValueError("Unknown distribution.")

def zero_preserved_log_stats(X):
    Y = np.copy(X)
    is_zero = (Y == 0)
    Y[is_zero] = np.nan
    Y_log = np.log(Y)
    nonzero_mean = np.nanmean(Y_log, axis=0, keepdims=True)
    nonzero_std = np.nanstd(Y_log, axis=0, keepdims=True)
    return nonzero_mean, nonzero_std

def zero_preserved_log_normalize(X, nonzero_mean, nonzero_std, log_output=False, zero_id=-3, shift=1.0):
    Y = np.copy(X)
    is_zero = (Y == 0)
    Y[is_zero] = np.nan
    Y_log = np.log(Y)
    Y_log = (Y_log-nonzero_mean)/nonzero_std + shift
    if log_output: Y = Y_log
    else: Y = np.exp(Y_log)
    Y[is_zero] = zero_id
    return Y

def zero_preserved_log_denormalize(Y, nonzero_mean, nonzero_std, log_input=False, zero_id=-3, shift=1.0):
    X = np.copy(Y)
    is_zero = (X == zero_id)
    X[is_zero] = np.nan
    if log_input: X_log = X
    else: X_log = np.log(X)
    X_log = (X_log-shift)*nonzero_std + nonzero_mean
    X = np.exp(X_log)
    X[is_zero] = 0
    return X

class CircularTransformer():
    def __init__(self, max_conds=None, min_conds=None):
        self.max_conds, self.min_conds = max_conds, min_conds

    def fit_transform(self, data):
        if self.max_conds is None: self.max_conds = np.max(data,axis=0,keepdims=False)
        if self.min_conds is None: self.min_conds = np.min(data,axis=0,keepdims=False)
        angles = (data-self.min_conds)/(self.max_conds-self.min_conds+1)*2*np.pi
        return np.concatenate((np.cos(angles),np.sin(angles)),axis=1)
    
    def transform(self, data):
        angles = (data-self.min_conds)/(self.max_conds-self.min_conds+1)*2*np.pi
        return np.concatenate((np.cos(angles),np.sin(angles)),axis=1)

    def inverse_transform(self, data):
        return (np.arctan2(data[:,1],data[:,0])/(2*np.pi)*(self.max_conds-self.min_conds+1)+self.min_conds)
    
class DirichletTransformer():
    def __init__(self, num_dims=None, transform_style="sample"):
        self.transform_style = transform_style
        self.num_dims = num_dims

    def transform(self, data):
        if self.transform_style == "sample": 
            gamma_rvs = np.random.gamma(data)
            return (gamma_rvs/gamma_rvs.sum(1, keepdims=True))
        elif self.transform_style == "embed": 
            embedding = np.zeros(self.num_dims+1)
            embedding[-1] = data.sum()
            embedding[:-1] = data/embedding[-1]
            return embedding
        else:
            raise NotImplementedError
        
class MinMaxTransformer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def fit(self, X):
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / self.data_range_
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        return self

    def transform(self, X):
        X_scaled = X * self.scale_ + self.min_
        X_clipped = np.clip(X_scaled, self.feature_range[0], self.feature_range[1])
        return X_clipped

    def inverse_transform(self, X):
        X_inversed = (X - self.min_) / self.scale_
        X_inversed_clipped = np.clip(X_inversed, self.data_min_, self.data_max_)
        return X_inversed_clipped

    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
class EarlyStopping:
    def __init__(self, patience=5, delta=0.1):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, score): 
        if self.best_score is None: self.best_score = score

        elif score <= (self.best_score + self.delta) and not self.early_stop:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: 
                self.early_stop = True
                print("Early stopping initiated.")

        elif score > (self.best_score + self.delta) and not self.early_stop:
            self.best_score = score
            self.counter = 0
            print(f"New (significant) best score: {self.best_score:5e}")

def find_matching_model(base_dir, user_config_dict):
    for subdir, _, _ in os.walk(base_dir):
        model_path = os.path.join(subdir, 'user_config_dict.json')
        if os.path.exists(model_path):
            with open(model_path, 'r') as f: saved_user_config_dict = json.load(f)
            if saved_user_config_dict==user_config_dict: return subdir
    return None

def get_latest_path(base_dir):
    paths = [os.path.join(base_dir,f) for f in os.listdir(base_dir)]
    latest_path = paths[np.argmax([os.path.getmtime(f) for f in paths])]
    return latest_path