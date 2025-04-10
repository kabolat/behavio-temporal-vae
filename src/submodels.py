import torch
from .utils import *

ACTIVATION = torch.nn.ReLU()

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

        if type(output_dim) is int:
            for param in dist_params:
                self.block_dict[param] = NNBlock(num_neurons, output_dim, num_neurons=num_neurons, num_hidden_layers=0, dropout=dropout, dropout_rate=dropout_rate, batch_normalization=batch_normalization)
        elif type(output_dim) is list:
            for i, param in zip(range(len(output_dim)), dist_params):
                self.block_dict[param] = NNBlock(num_neurons, output_dim[i], num_neurons=num_neurons, num_hidden_layers=0, dropout=dropout, dropout_rate=dropout_rate, batch_normalization=batch_normalization)
        elif type(output_dim) is dict:
            for key in output_dim.keys():
                self.block_dict[key] = NNBlock(num_neurons, output_dim[key], num_neurons=num_neurons, num_hidden_layers=0, dropout=dropout, dropout_rate=dropout_rate, batch_normalization=batch_normalization)

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
    def __init__(self, input_dim, output_dim, sigma_fixed=1.0, sigma_lim=0.1, marginal_std_lim=None, average_max_std=1.0, mu_upper_lim=5.0, mu_lower_lim=-3.0, learn_sigma=True, num_hidden_layers=2, num_neurons=50, dropout=True, dropout_rate=0.5, batch_normalization=True, **_):
        super(GaussianNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learn_sigma = learn_sigma
        self.sigma_fixed = sigma_fixed
        if marginal_std_lim is not None: 
            self.sigma_lower_lim = marginal_std_lim
            print("USING MARGINAL_STD_LIM!")
        else:
            self.sigma_lower_lim = sigma_lim
            print("USING SIGMA_LIM!")
        self.sigma_upper_lim = average_max_std
        self.mu_upper_lim = mu_upper_lim
        self.mu_lower_lim = mu_lower_lim

        if learn_sigma: dist_params = ["mu", "sigma"]
        else: dist_params = ["mu"]

        self.parameterizer = ParameterizerNN(input_dim, output_dim, dist_params=dist_params, num_hidden_layers=num_hidden_layers, num_neurons=num_neurons, dropout=dropout, dropout_rate=dropout_rate, batch_normalization=batch_normalization)

    def _num_parameters(self):
        return self.parameterizer._num_parameters()
    
    def forward(self, inputs):
        param_dict = self.parameterizer(inputs)
        param_dict["mu"] = param_dict["mu"].clamp(self.mu_lower_lim, self.mu_upper_lim)
        if self.learn_sigma: 
            param_dict["sigma"] = to_sigma(param_dict["sigma"]).clamp(self.sigma_lower_lim, self.sigma_upper_lim)
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

    def create_covariance_matrix(self, param_dict):
        return torch.eye(param_dict["mu"].shape[-1], device=param_dict["mu"].device).expand(*param_dict["sigma"].shape[:-1], -1, -1) * param_dict["sigma"][..., None]**2
    
    def get_marginal_sigmas(self, param_dict):
        return param_dict["sigma"]

class DictionaryGaussian(torch.nn.Module):
    def __init__(self, input_dim, output_dim, vocab_size=100, sigma_lim=0.1, marginal_std_lim=0.01, average_max_std=1.0, mu_upper_lim=5.0, mu_lower_lim=-3.0, num_hidden_layers=2, num_neurons=50, dropout=True, dropout_rate=0.5, batch_normalization=True, **_):
        super(DictionaryGaussian, self).__init__()
        
        if vocab_size <= output_dim: raise ValueError("Vocabulary size should be larger than the output dimension. Please increase the vocabulary size.")
        self.vocab_size = vocab_size

        self.marginal_std_lim = marginal_std_lim
        self.output_dim = output_dim
        self.mu_upper_lim = mu_upper_lim
        self.mu_lower_lim = mu_lower_lim
        self.sigma_lower_lim = sigma_lim
        self.sigma_upper_lim = (output_dim/self.vocab_size)**.5 * (average_max_std - self.marginal_std_lim)
        if self.sigma_upper_lim < self.sigma_lower_lim: raise ValueError("Sigma upper limit is smaller than sigma lower limit. Please decresase the vocabulary size or the sigma lower limit.")

        dist_params = ["mu", "sigma"]
        output_dim_dict = {"mu":output_dim, "sigma":vocab_size}

        self.parameterizer = ParameterizerNN(input_dim, output_dim_dict, dist_params=dist_params, num_hidden_layers=num_hidden_layers, num_neurons=num_neurons, dropout=dropout, dropout_rate=dropout_rate, batch_normalization=batch_normalization)

        self.SigmaMapper = torch.nn.Parameter(torch.randn(output_dim, vocab_size), requires_grad=True)
    
    def _num_parameters(self):
        return self.parameterizer._num_parameters()

    def get_SigmaMapper(self):
        return matrix_normalizer(self.SigmaMapper)
    
    def forward(self, inputs):
        param_dict = self.parameterizer(inputs)
        param_dict["sigma"] = torch.nn.functional.relu(param_dict["sigma"]).clamp(self.sigma_lower_lim, self.sigma_upper_lim)
        param_dict["mu"] = param_dict["mu"].clamp(self.mu_lower_lim, self.mu_upper_lim)
        return param_dict
    
    def rsample(self, param_dict=None, num_samples=1, **_):
        mu, sigma = param_dict["mu"], param_dict["sigma"]
        eps = torch.randn((num_samples, *sigma.shape), device=sigma.device)
        corr_sigma = self.get_SigmaMapper() @ (eps*sigma)[...,None]
        return mu + corr_sigma.squeeze(-1)
    
    def sample(self, param_dict=None, num_samples=1, **_):
        return self.rsample(param_dict=param_dict, num_samples=num_samples)
    
    def log_likelihood(self, targets, param_dict=None):
        Sigma = self.create_covariance_matrix(param_dict)
        return torch.distributions.MultivariateNormal(param_dict["mu"], covariance_matrix=Sigma).log_prob(targets)
    
    def kl_divergence(self, param_dict=None, prior_params={"mu":0.0, "sigma":1.0}):
        pass

    def create_covariance_matrix(self, param_dict):
        U = self.get_SigmaMapper()
        S = U*param_dict["sigma"][...,None,:]
        return (S @ S.mT) + torch.eye(self.output_dim, device=param_dict["mu"].device)*(self.marginal_std_lim**2)
    
    def get_marginal_sigmas(self, param_dict):
        return torch.sqrt(torch.diagonal(self.create_covariance_matrix(param_dict), dim1=-2, dim2=-1))

def get_distribution_model(dist_type, **kwargs):
    if dist_type.lower() in ["gaussian", "gauss", "normal", "n", "g"]: return GaussianNN(**kwargs)
    elif dist_type.lower() in ["dictionary-gaussian", "dict-gauss"]: return DictionaryGaussian(**kwargs)
    else: raise NotImplementedError("Unknown distribution type: {}".format(dist_type))

def get_prior_params(dist_type, num_dims=1):
    is_list = type(dist_type) is list
    if not is_list: 
        dist_type, num_dims = [dist_type], [num_dims]
    params = ['']*len(dist_type)
    for i in range(len(dist_type)):
        if dist_type[i].lower() in ["gaussian", "gauss", "normal", "n", "g"]: params[i] = {"mu":torch.zeros(num_dims[i]), "sigma":torch.ones(num_dims[i])}
        else: raise NotImplementedError("Unknown distribution type: {}".format(dist_type[i]))
    if not is_list: return params[0]
    else: return params