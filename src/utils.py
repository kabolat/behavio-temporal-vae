import torch
from torch.nn.functional import softplus
from torch import inf
import torch
from pandas import Timestamp
from datetime import timedelta
import numpy as np
import os, json

def to_positive(xtilde): return torch.nn.functional.softplus(xtilde,beta=1,threshold=5)

def from_positive(x): return x + torch.log(-torch.expm1(-x))

def to_sigma(sigmatilde):       return to_positive(sigmatilde)

def from_sigma(sigma):      return from_positive(sigma)

def to_alpha(alphatilde):       return to_positive(alphatilde)

def from_alpha(alpha):      return from_positive(alpha)

def to_pi(pitilde):     return torch.sigmoid(torch.clip(pitilde,-10,10))

def from_pi(pi):    return torch.logit(pi,eps=1e-3)

def alpha_to_mu(alpha):
    if alpha.ndim != 2: raise ValueError("alpha must be a 2D tensor.")
    num_dims = alpha.shape[1]
    if num_dims < 3: raise ValueError("num_dims must be at least 3 for this implementation.")
    log_alpha = alpha.log()
    return log_alpha - 1/num_dims * log_alpha.sum(dim=1, keepdim=True)

def alpha_to_sigma(alpha):
    if alpha.ndim != 2: raise ValueError("alpha must be a 2D tensor.")
    num_dims = alpha.shape[1]
    if num_dims < 3: raise ValueError("num_dims must be at least 3 for this implementation.")
    return torch.sqrt(1/alpha * (1 - 2/num_dims) + 1/num_dims**2 * (1/alpha).sum(dim=1, keepdim=True))

def mu_sigma_to_alpha(mu, sigma):
    if mu.ndim != 2 or sigma.ndim != 2: raise ValueError("mu and sigma must be 2D tensors.")
    num_dims = mu.shape[1]
    if num_dims < 3: raise ValueError("num_dims must be at least 3 for this implementation.")
    return 1/sigma**2 * (1 - 2/num_dims + torch.exp(-mu)/num_dims**2 * torch.exp(-mu).sum(dim=1, keepdim=True))

def matrix_normalizer(matrix):
    return matrix / torch.linalg.norm(matrix, dim=0, ord=2, keepdim=True)

def KMSMatrix(rho, num_dims, typ=None):
    if typ is None or typ == "self":
        # zeta = torch.pow(rho, torch.arange(num_dims, device=rho.device))
        # L = (zeta[...,None] / zeta[...,None,:]).tril(-1)
        # return L+L.mT+torch.eye(num_dims, device=rho.device)
        pows = torch.arange(num_dims, device=rho.device)
        t = toeplitz(pows,pows)
        return torch.pow(rho[...,None],t)
    elif typ == "inv" or typ == "inverse":
        R = torch.diagflat(torch.ones(num_dims-1, device=rho.device), offset=-1)*rho[...,None]
        return (torch.eye(num_dims, device=rho.device)-R-R.mT+R.roll(1, dims=-1)*R.roll(-1, dims=-2))/(1-rho[...,None]**2)
    elif typ == "chol" or typ == "cholesky":
        L = KMSMatrix(rho, num_dims, typ="self").tril(0)
        D = torch.ones(num_dims, device=rho.device)*(1-rho**2).sqrt()
        D[...,0] = 1
        return (torch.ones(L.shape,device=D.device)*D[...,None,:])*L

def toeplitz(c, r):
    vals = torch.cat((r, c[1:].flip(0)))
    shape = len(c), len(r)
    i, j = torch.ones(*shape).nonzero().T
    return vals[j-i].reshape(*shape)

def lower_toeplitz(l):
    vals = torch.cat((l, l[...,1:]*0), dim=-1)
    dim = l.shape[-1]
    i, j = torch.ones((dim,dim)).nonzero().T
    return vals[...,i-j].reshape(*list(l.shape),dim)

def log_anneal(step, num_steps, start_value, end_value):
    return start_value*(end_value/start_value)**(step/num_steps)

def _gen_gumbels(logits):
    gumbels = -torch.empty_like(logits).exponential_().log()
    if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
        # to avoid zero in exp output
        gumbels = _gen_gumbels(logits)
    return gumbels

def gumbel_softmax(logits, tau=1, hard=False, dim=-1):
    gumbels = _gen_gumbels(logits)  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

def log_prob(dist="Normal", params=None, targets=None):
    if dist=="Normal":
        return -0.5*torch.log(2*torch.tensor(torch.pi)) - torch.log(params["sigma"]) - 0.5*((targets-params["mu"])/params["sigma"])**2
    elif dist=="MultivariateNormal":
        ...
    elif dist=="Bernoulli":
        return targets*torch.log(params["pi"]) + (1-targets)*torch.log(1-params["pi"])
    elif dist=="Mixed":
        is_continuous = (targets!=0).float()
        return log_prob("Bernoulli", params, is_continuous) + log_prob("Normal", params, targets)*is_continuous
    elif dist=="Dirichlet":
        return (params["alpha"]-1)*torch.log(targets)
    elif dist=="Categorical":
        return (torch.log(params["pi"])*targets).sum(1) #one-hot targets
    else:
        raise ValueError("Unknown distribution.")

def kl_divergence(dist="Normal", params=None, prior_params=None):
    if dist=="Normal":
        var_ratio = (params["sigma"] / prior_params["sigma"]).pow(2)
        t1 = ((params["mu"] - prior_params["mu"]) / prior_params["sigma"]).pow(2)
        return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
    elif dist=="Bernoulli":
        pi, logits = params["pi"], torch.logit(params["pi"])
        pi0, logits0 = prior_params["pi"].expand(pi.shape[0],-1), torch.logit(prior_params["pi"]).expand(pi.shape[0],-1)
        t1 = pi * (softplus(-logits0) - softplus(-logits))
        t1[pi0 == 0], t1[pi == 0] = inf, 0
        t2 = (1 - pi) * (softplus(logits0) - softplus(logits))
        t2[pi0 == 1], t2[pi == 1] = inf, 0
        return t1 + t2
    elif dist=="Mixed":
        return kl_divergence("Bernoulli", params, prior_params) + params["pi"] * kl_divergence("Normal", params, prior_params)
    elif dist=="Dirichlet":
        alpha_post, alpha_prior, K = params["alpha"], prior_params["alpha"], params["alpha"].shape[-1]
        return (torch.lgamma(alpha_post.sum(dim=-1,keepdim=True)) - torch.lgamma(alpha_prior.sum(dim=-1,keepdim=True)))/K + torch.lgamma(alpha_prior) - torch.lgamma(alpha_post) + ((alpha_post - alpha_prior) * (torch.digamma(alpha_post) - torch.digamma(alpha_post.sum(dim=-1,keepdim=True))))
    elif dist=="Categorical":
        return (params["pi"]*torch.log(params["pi"]/prior_params["pi"]+1e-8)).sum(1)
    else:
        raise ValueError("Unknown distribution.")

def joint_distribution(dist="normal", params=None):
    if dist=="normal":
        precisions = 1/params["sigma"]**2
        precision = precisions.sum(dim=0,keepdim=True)
        mu = torch.sum(params["mu"]*precisions,dim=0,keepdim=True)/precision
        return {"mu":mu,"sigma":torch.sqrt(1/precision)}
    else:
        raise NotImplementedError("Unknown distribution.")

def circlize(data):
    max_conds = torch.max(data,dim=0,keepdim=False)[0]
    min_conds = torch.min(data,dim=0,keepdim=False)[0]
    return torch.cat((torch.cos((data-min_conds)/(max_conds-min_conds+1)*2*torch.pi),torch.sin((data-min_conds)/(max_conds-min_conds+1)*2*torch.pi)),dim=1)

def get_timeslots(delta):
    timeslots=[]
    my_day = Timestamp('2012')
    for _ in range(int(24*60/delta)):
        timeslots.append(my_day.strftime('%H:%M'))
        my_day+=timedelta(minutes=delta)
    return timeslots

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

def find_matching_model(base_dir, model_kwargs, fit_kwargs):
    for subdir, _, _ in os.walk(base_dir):
        model_kwargs_path = os.path.join(subdir, 'model_kwargs.json')
        fit_kwargs_path = os.path.join(subdir, 'fit_kwargs.json')
        model_path = os.path.join(subdir, 'model.pkl')
        if os.path.exists(fit_kwargs_path) and os.path.exists(model_path) and os.path.exists(model_kwargs_path):
            with open(model_kwargs_path, 'r') as f: saved_model_kwargs = json.load(f)
            with open(fit_kwargs_path, 'r') as f: saved_fit_kwargs = json.load(f)
            if saved_model_kwargs==model_kwargs and saved_fit_kwargs == fit_kwargs: return subdir
    return None