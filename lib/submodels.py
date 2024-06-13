import torch
from torchrl.modules import OneHotCategorical
from .utils import *

ACTIVATION = torch.nn.Softplus()

class NNBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_neurons=50, num_hidden_layers=2, dropout=False, dropout_rate=0.5, batch_normalization=False, resnet=True, **_):
        super(NNBlock, self).__init__()
        self.input_dim   = input_dim
        self.output_dim  = output_dim
        self.num_neurons = num_neurons
        self.num_layers  = num_hidden_layers
        self.resnet      = resnet

        self.input_layer   = torch.nn.Sequential(torch.nn.Linear(input_dim, num_neurons), ACTIVATION)
        self.middle_layers = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(num_neurons, num_neurons), ACTIVATION) for _ in range(num_hidden_layers)])
        self.output_layer  = torch.nn.Sequential(torch.nn.Linear(num_neurons, output_dim))

        if batch_normalization:
            self.input_layer.append(torch.nn.BatchNorm1d(num_neurons))
            for i in range(len(self.middle_layers)): self.middle_layers[i].append(torch.nn.BatchNorm1d(num_neurons))
        
        if dropout:
            self.input_layer.append(torch.nn.Dropout(dropout_rate))
            for i in range(len(self.middle_layers)): self.middle_layers[i].append(torch.nn.Dropout(dropout_rate))


    def forward(self, x):
        h = x.view(-1, self.input_dim)
        h = self.input_layer(h)
        for layer in self.middle_layers:
            if self.resnet: h = layer(h)+h
            else: h = layer(h)
        h = self.output_layer(h)
        return h

    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ParameterizerNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dist_params=["mu"], num_hidden_layers=2, num_neurons=50, dropout=False, dropout_rate=0.5, batch_normalization=False, **_):
        super(ParameterizerNN, self).__init__()
        self.dist_params = dist_params
        self.block_dict = torch.nn.ModuleDict()

        self.block_dict["input"] = NNBlock(input_dim, num_neurons, num_neurons=num_neurons, num_hidden_layers=num_hidden_layers, dropout=dropout, dropout_rate=dropout_rate, batch_normalization=batch_normalization)
        
        self.block_dict["input"].output_layer.append(ACTIVATION)
        if batch_normalization: self.block_dict["input"].output_layer.append(torch.nn.BatchNorm1d(num_neurons))
        if dropout: self.block_dict["input"].output_layer.append(torch.nn.Dropout(dropout_rate))

        for param in dist_params:
            self.block_dict[param] = NNBlock(num_neurons, output_dim, num_neurons=num_neurons, num_hidden_layers=1, dropout=dropout, dropout_rate=dropout_rate, batch_normalization=batch_normalization)
            # if batch_normalization: self.block_dict[param].output_layer.append(torch.nn.BatchNorm1d(output_dim))
            # if dropout: self.block_dict[param].output_layer.append(torch.nn.Dropout(dropout_rate))

    def forward(self, inputs):
        h = inputs.view(-1, self.block_dict["input"].input_dim)
        h = self.block_dict["input"](h)
        output_dict = {}
        for param in self.dist_params:
            output_dict[param] = self.block_dict[param](h)
        return output_dict
    
    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class GaussianNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, sigma_fixed=1.0, sigma_lim=0.1, learn_sigma=True, num_hidden_layers=2, num_neurons=50, dropout=True, dropout_rate=0.5, batch_normalization=True, **_):
        super(GaussianNN, self).__init__()

        self.learn_sigma = learn_sigma
        self.sigma_fixed = sigma_fixed
        self.sigma_lim = sigma_lim

        if learn_sigma: dist_params = ["mu", "sigma"]
        else: dist_params = ["mu"]

        self.parameterizer = ParameterizerNN(input_dim, output_dim, dist_params=dist_params, num_hidden_layers=num_hidden_layers, num_neurons=num_neurons, dropout=dropout, dropout_rate=dropout_rate, batch_normalization=batch_normalization)

    def _num_parameters(self):
        return self.parameterizer._num_parameters()
    
    def forward(self, inputs):
        param_dict = self.parameterizer(inputs)
        if self.learn_sigma: param_dict["sigma"] = to_sigma(param_dict["sigma"]).clamp(self.sigma_lim, None)
        else: param_dict["sigma"] = torch.ones_like(param_dict["mu"])*self.sigma_fixed
        return param_dict
    
    def rsample(self, param_dict=None, num_samples=1, **_):
        return param_dict["mu"] + param_dict["sigma"] * torch.randn((num_samples, *param_dict["mu"].shape), device=param_dict["mu"].device)

    def sample(self, param_dict=None, num_samples=1, **_):
        return self.rsample(param_dict=param_dict, num_samples=num_samples)
    
    def log_likelihood(self, targets, param_dict=None):
        return log_prob("Normal", param_dict, targets).sum(-1)
    
    def kl_divergence(self, param_dict=None, prior_params={"mu":0.0, "sigma":1.0}):
        return kl_divergence("Normal", param_dict, prior_params)
    

class KMSGaussianNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, sigma_fixed=1.0, sigma_lim=0.1, learn_sigma=True, num_hidden_layers=2, num_neurons=50, dropout=True, dropout_rate=0.5, batch_normalization=True, **_):
        super(KMSGaussianNN, self).__init__()

        self.learn_sigma = learn_sigma
        self.sigma_fixed = sigma_fixed
        self.sigma_lim = sigma_lim

        dist_params = ["mu", "sigma", "tau"]

        self.parameterizer = ParameterizerNN(input_dim, output_dim, dist_params=dist_params, num_hidden_layers=num_hidden_layers, num_neurons=num_neurons, dropout=dropout, dropout_rate=dropout_rate, batch_normalization=batch_normalization)

    def _num_parameters(self):
        return self.parameterizer._num_parameters()
    
    def forward(self, inputs):
        param_dict = self.parameterizer(inputs)
        param_dict["sigma"] = to_sigma(param_dict["sigma"]).clamp(self.sigma_lim, None)
        tau = self.get_tau(param_dict["tau"])
        param_dict["tau"] = param_dict["tau"]*0 + tau   ##repeated tau's for modularity with VAE
        return param_dict
    
    def tau2rho(self, tau): return torch.exp(-1/torch.abs(tau)) * torch.sign(tau)

    def get_tau(self, tensor_tau): return tensor_tau[...,[0]]
    
    def create_covariance_matrix(self, param_dict):
        sigma, tau = param_dict["sigma"], self.get_tau(param_dict["tau"])
        return KMSMatrix(self.tau2rho(tau), sigma.shape[-1], typ="self") * (sigma[...,None]*sigma[...,None,:])
    
    def create_precision_matrix(self, param_dict):
        sigma, tau = param_dict["sigma"], self.get_tau(param_dict["tau"])
        return KMSMatrix(self.tau2rho(tau), sigma.shape[-1], typ="inv") / (sigma[...,None]*sigma[...,None,:])
    
    def create_scale_tril_matrix(self, param_dict):
        sigma, tau = param_dict["sigma"], self.get_tau(param_dict["tau"])
        m_chol = KMSMatrix(self.tau2rho(tau), sigma.shape[-1], typ="chol")
        return (torch.ones(m_chol.shape,device=sigma.device)*sigma[...,None])*m_chol

    def rsample(self, param_dict=None, num_samples=1, **_):
        L = self.create_scale_tril_matrix(param_dict)
        return torch.distributions.MultivariateNormal(param_dict["mu"], scale_tril=L).rsample((num_samples,))

    def sample(self, param_dict=None, num_samples=1, **_):
        L = self.create_scale_tril_matrix(param_dict)
        return torch.distributions.MultivariateNormal(param_dict["mu"], scale_tril=L).sample((num_samples,))
    
    def log_likelihood(self, targets, param_dict=None):
        L = self.create_scale_tril_matrix(param_dict)
        return torch.distributions.MultivariateNormal(param_dict["mu"], scale_tril=L).log_prob(targets)
    
    def kl_divergence(self, param_dict=None, prior_params={"mu":0.0, "sigma":1.0}):
        pass


class NonDiagonalGaussianNN(KMSGaussianNN):
    def __init__(self, input_dim, output_dim, sigma_fixed=1.0, sigma_lim=0.1, learn_sigma=True, num_hidden_layers=2, num_neurons=50, dropout=True, dropout_rate=0.5, batch_normalization=True, **_):
        super(NonDiagonalGaussianNN, self).__init__(input_dim, output_dim, sigma_fixed=sigma_fixed, sigma_lim=sigma_lim, learn_sigma=learn_sigma, num_hidden_layers=num_hidden_layers, num_neurons=num_neurons, dropout=dropout, dropout_rate=dropout_rate, batch_normalization=batch_normalization)

        dist_params = ["mu", "sigma", "tau", "eta"]

        self.parameterizer = ParameterizerNN(input_dim, output_dim, dist_params=dist_params, num_hidden_layers=num_hidden_layers, num_neurons=num_neurons, dropout=dropout, dropout_rate=dropout_rate, batch_normalization=batch_normalization)
    
    def forward(self, inputs):
        param_dict = self.parameterizer(inputs)
        param_dict["sigma"] = to_sigma(param_dict["sigma"]).clamp(self.sigma_lim, None)
        tau = self.get_tau(param_dict["tau"])
        eta = self.get_eta(param_dict["eta"])
        param_dict["tau"] = param_dict["tau"]*0 + tau
        param_dict["eta"] = param_dict["eta"]*0 + eta
        return param_dict
    
    def get_tau(self, tensor_tau):
        tensor_tau = (tensor_tau).tanh()*5e-1
        tensor_tau[...,0] = tensor_tau[...,0]*0 + 1
        return tensor_tau

    def get_eta(self, tensor_eta): return (tensor_eta).tanh()*5e-1

    def get_unnormalized_cholesky_matrix(self, param_dict):
        tau, eta = param_dict["tau"], param_dict["eta"]
        m_chol = lower_toeplitz(tau)
        m_chol = (m_chol * eta[...,None,:])
        eye_mask = torch.eye(m_chol.shape[-1], device=m_chol.device)
        m_chol = m_chol * (1-eye_mask) + eye_mask
        return m_chol
    
    def get_cholesky_matrix(self, param_dict):
        m_chol = self.get_unnormalized_cholesky_matrix(param_dict)
        return m_chol / torch.linalg.vector_norm(m_chol, dim=-1, keepdim=True)

    def create_covariance_matrix(self, param_dict):
        L = self.create_scale_tril_matrix(param_dict)
        return torch.matmul(L, L.transpose(-1,-2))

    def create_precision_matrix(self, param_dict):
        return torch.inverse(self.create_covariance_matrix(param_dict))

    def create_scale_tril_matrix(self,param_dict):
        m_chol = self.get_cholesky_matrix(param_dict)
        return (torch.ones(m_chol.shape,device=m_chol.device)*param_dict["sigma"][...,None])*m_chol


class LogitNormalNN(GaussianNN):
    def __init__(self, input_dim, output_dim, conv=False, num_hidden_layers=2, num_neurons=50, dropout=True, dropout_rate=0.5, batch_normalization=True, **_):
        super(LogitNormalNN, self).__init__(input_dim=input_dim, output_dim=output_dim, num_hidden_layers=num_hidden_layers, num_neurons=num_neurons, learn_sigma=True, dropout=True, dropout_rate=0.5, batch_normalization=True)

    def _num_parameters(self): return super().parameterizer._num_parameters()
    
    def sample(self, param_dict=None, num_samples=1, **_): return super().sample(param_dict=param_dict, num_samples=num_samples).sigmoid()

    def rsample(self, param_dict=None, num_samples=1, **_): return super().rsample(param_dict=param_dict, num_samples=num_samples).sigmoid()
    
    def log_likelihood(self, targets, param_dict=None): return super().log_likelihood(targets.logit(), param_dict).sum(-1)

class DirichletNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, conv=False, num_hidden_layers=2, num_neurons=50, dropout=True, dropout_rate=0.5, batch_normalization=True, **_):
        super(DirichletNN, self).__init__()

        dist_params = ["alpha"]

        self.parameterizer = ParameterizerNN(input_dim, output_dim, dist_params=dist_params, num_hidden_layers=num_hidden_layers, num_neurons=num_neurons, dropout=dropout, dropout_rate=dropout_rate, batch_normalization=batch_normalization)

    def _num_parameters(self): return self.parameterizer._num_parameters()
    
    def forward(self, inputs, constrain=True):
        total_count_per_word = inputs.sum(dim=-1, keepdim=True)
        param_dict = self.parameterizer(inputs/total_count_per_word)
        if constrain: 
            param_dict["alpha"] = to_alpha(param_dict["alpha"])
            param_dict["alpha"] = param_dict["alpha"]*total_count_per_word
        return param_dict
    
    def sample(self, param_dict=None, num_samples=1, **_):
        return torch.distributions.Dirichlet(param_dict["alpha"]).sample((num_samples,))

    def rsample(self, param_dict=None, num_samples=1, approximate=False, **_):
        if approximate:
            mu, sigma = alpha_to_mu(param_dict["alpha"]), alpha_to_sigma(param_dict["alpha"])
            samples = mu + sigma * torch.randn((num_samples, *mu.shape))
            return samples.softmax(dim=-1)
        else:
            return torch.distributions.Dirichlet(param_dict["alpha"]).rsample((num_samples,))
    
    def log_likelihood(self, targets, param_dict=None):
        return log_prob("Dirichlet", param_dict, targets).sum(-1)
    
    def kl_divergence(self, param_dict=None, prior_params={"alpha":0.5}):
        return kl_divergence("Dirichlet", param_dict, prior_params)

class BetaDecoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, temperature=1.0, prodlda=False, **_):
        super(BetaDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prodlda = prodlda
        self.beta_unnorm = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.temperature = temperature

    def forward(self, x, temperature=None):
        if temperature is None: temperature = self.temperature
        if self.prodlda:
            return (torch.matmul(x, self.beta_unnorm)/temperature).softmax(dim=-1)
        else:
            beta = self.get_beta(temperature=temperature)
            return torch.matmul(x, beta)
        
    def _num_parameters(self):
        return self.beta_unnorm.numel()
    
    def get_beta(self, temperature=None):
        if temperature is None: temperature = self.temperature
        return (self.beta_unnorm/temperature).softmax(dim=-1)


class BernoulliNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers=2, num_neurons=50, **_):
        super(BernoulliNN, self).__init__()

        dist_params = ["pi"]

        self.parameterizer = ParameterizerNN(input_dim, output_dim, dist_params=dist_params, num_hidden_layers=num_hidden_layers, num_neurons=num_neurons)

    def _num_parameters(self):
        return self.parameterizer._num_parameters()
    
    def forward(self, inputs):
        param_dict = self.parameterizer(inputs)
        param_dict["pi"] = to_pi(param_dict["pi"])
        return param_dict
    
    def sample(self, param_dict=None, num_samples=1, **_):
        return torch.bernoulli(param_dict["pi"].expand(num_samples, *param_dict["pi"].shape))

    def rsample(self, param_dict=None, num_samples=1, tau=1, hard=True, **_):
        pi_1 = param_dict["pi"].unsqueeze(0).expand(num_samples, *param_dict["pi"].shape)
        pi_0 = 1 - pi_1
        logits = torch.stack([torch.log(pi_0), torch.log(pi_1)], dim=0)
        samples = gumbel_softmax(logits, tau=tau, hard=hard, dim=0)[1]
        return samples
    
    def log_likelihood(self, targets, param_dict=None):
        return log_prob("Bernoulli", param_dict, targets).sum(-1)
    
    def kl_divergence(self, param_dict=None, prior_params={"pi":0.5}):
        return kl_divergence("Bernoulli", param_dict, prior_params)

class CategoricalNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers=2, num_neurons=50, **_):
        super(CategoricalNN, self).__init__()

        dist_params = ["pi"]

        self.parameterizer = ParameterizerNN(input_dim, output_dim, dist_params=dist_params, num_hidden_layers=num_hidden_layers, num_neurons=num_neurons)

    def _num_parameters(self):
        return self.parameterizer._num_parameters()
    
    def forward(self, inputs):
        param_dict = self.parameterizer(inputs)
        param_dict["pi"] = param_dict["pi"].softmax(-1)
        return param_dict
    
    def sample(self, param_dict=None, num_samples=1, **_):
        return OneHotCategorical(probs=param_dict["pi"]).sample((num_samples,))

    def rsample(self, param_dict=None, num_samples=1, **_):
        return OneHotCategorical(probs=param_dict["pi"]).rsample((num_samples,))
    
    def log_likelihood(self, targets, param_dict=None):
        return log_prob("Bernoulli", param_dict, targets).sum(-1)
    
    def kl_divergence(self, param_dict=None, prior_params={"pi":0.5}):
        return kl_divergence("Bernoulli", param_dict, prior_params)

class MixedNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, continuous_dist_type = "normal", learn_sigma=True, num_hidden_layers=2, num_neurons=50, **_):
        super(MixedNN, self).__init__()

        self.discerete_dist = BernoulliNN(input_dim, output_dim, num_hidden_layers=1, num_neurons=num_neurons)
        self.continuous_dist = get_distribution_model(continuous_dist_type, input_dim=input_dim, output_dim=output_dim, learn_sigma=learn_sigma, num_hidden_layers=num_hidden_layers, num_neurons=num_neurons, **_)

    def _num_parameters(self):
        return self.discerete_dist._num_parameters() + self.continuous_dist._num_parameters()
    
    def forward(self, inputs):
        param_dict = {}
        param_dict.update(self.discerete_dist(inputs))
        param_dict.update(self.continuous_dist(inputs))
        return param_dict
    
    def sample(self, param_dict=None, num_samples=1, **_):
        mask = self.discerete_dist.sample(param_dict=param_dict, num_samples=num_samples)
        samples = self.continuous_dist.sample(param_dict=param_dict, num_samples=num_samples)
        return mask*samples
    
    def rsample(self, param_dict=None, num_samples=1, tau=1, **_):
        mask = self.discerete_dist.rsample(param_dict=param_dict, num_samples=num_samples, tau=tau, hard=True)
        samples = self.continuous_dist.rsample(param_dict=param_dict, num_samples=num_samples)
        return mask*samples
    
    def log_likelihood(self, targets, param_dict=None):
        return log_prob("Mixed", param_dict, targets).sum(-1)
    
    def kl_divergence(self, param_dict=None, prior_params={"pi":0.5,"mu":0.0,"sigma":1.0}):
        return kl_divergence("Mixed", param_dict, prior_params)

def get_distribution_model(dist_type, **kwargs):
    if dist_type.lower() in ["gaussian", "gauss", "normal", "n", "g"]: return GaussianNN(**kwargs)
    elif dist_type.lower() in ["kms-gaussian", "kms"]: return KMSGaussianNN(**kwargs)
    elif dist_type.lower() in ["non-diagonal-gaussian", "ndg"]: return NonDiagonalGaussianNN(**kwargs)
    elif dist_type.lower() in ["dirichlet", "dir", "d"]: return DirichletNN(**kwargs)
    elif dist_type.lower() in ["logitnormal", "ln"]: return LogitNormalNN(**kwargs)
    elif dist_type.lower() in ["bernoulli", "bern", "b"]: return BernoulliNN(**kwargs)
    elif dist_type.lower() in ["mixed", "mix", "m"]: return MixedNN(**kwargs)
    else: raise NotImplementedError("Unknown distribution type: {}".format(dist_type))

def get_prior_params(dist_type, num_dims=1):
    is_list = type(dist_type) is list
    if not is_list: 
        dist_type, num_dims = [dist_type], [num_dims]
    params = ['']*len(dist_type)

    for i in range(len(dist_type)):
        if dist_type[i].lower() in ["gaussian", "gauss", "normal", "n", "g"]: params[i] = {"mu":torch.zeros(num_dims[i]), "sigma":torch.ones(num_dims[i])}
        elif dist_type[i].lower() in ["bernoulli", "bern", "b"]: params[i] =  {"pi":torch.ones(num_dims[i])/2}
        elif dist_type[i].lower() in ["dirichlet", "dir", "d"]: params[i] = {"alpha":torch.ones(num_dims[i])}
        elif dist_type[i].lower() in ["logitnormal", "ln"]: params[i] = get_prior_params("gaussian", num_dims[i])
        elif dist_type[i].lower() in ["mixed", "mix", "m"]:
            params[i] = get_prior_params("bernoulli", num_dims[i])
            params[i].update(get_prior_params("gaussian", num_dims[i]))
        else: raise NotImplementedError("Unknown distribution type: {}".format(dist_type[i]))
    
    if not is_list: return params[0]
    else: return params