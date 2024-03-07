import torch
from torch.nn.functional import softplus
from torch import inf
import torch
from pandas import Timestamp
from datetime import timedelta
import numpy as np

def to_positive(xtilde): return torch.nn.functional.softplus(xtilde,beta=1,threshold=5)

def from_positive(x): return x + torch.log(-torch.expm1(-x))

def to_sigma(sigmatilde):       return to_positive(sigmatilde)

def from_sigma(sigma):      return from_positive(sigma)

def to_alpha(alphatilde):       return to_positive(alphatilde) + 1

def from_alpha(alpha):      return from_positive(alpha - 1)

def to_pi(pitilde):     return torch.sigmoid(torch.clip(pitilde,-10,10))

def from_pi(pi):    return torch.logit(pi,eps=1e-3)

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
    elif dist=="Bernoulli":
        return targets*torch.log(params["pi"]) + (1-targets)*torch.log(1-params["pi"])
    elif dist=="Mixed":
        is_continuous = (targets!=0).float()
        return log_prob("Bernoulli", params, is_continuous) + log_prob("Normal", params, targets)*is_continuous
    elif dist=="Dirichlet":
        return (params["alpha"]-1)*torch.log(targets)
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
    Y = X.clone()
    is_zero = (Y == 0)
    Y[is_zero] = torch.nan
    Y_log = torch.log(Y)
    nonzero_mean = torch.nanmean(Y_log, axis=0, keepdims=True)
    nonzero_std = torch.tensor(np.nanstd(Y_log, axis=0, keepdims=True))
    return nonzero_mean, nonzero_std

def zero_preserved_log_normalize(X, nonzero_mean, nonzero_std, shift=1.0):
    Y = X.clone()
    is_zero = (Y == 0)
    Y[is_zero] = torch.nan
    Y_log = torch.log(Y)
    Y_log = (Y_log-nonzero_mean)/nonzero_std + shift
    Y_log[is_zero] = 0
    return Y_log

def zero_preserved_log_denormalize(Y_log, nonzero_mean, nonzero_std, shift=1.0):
    X = Y_log.clone()
    is_zero = (X == 0)
    X[is_zero] = torch.nan
    X = (X-shift)*nonzero_std + nonzero_mean
    X = torch.exp(X)
    X[is_zero] = 0
    return X

# def to_alpha(mu, sigma):
#     K = mu.shape[-1]
#     return 1/sigma**2 * (1 - 2/K + torch.exp(-mu)/K**2 * torch.exp(-mu).sum(-1,keepdim=True))
