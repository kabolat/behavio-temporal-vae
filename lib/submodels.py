import torch
from .utils import *

ACTIVATION = torch.nn.Softplus()

class NNBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_neurons=50, num_hidden_layers=2, dropout=True, dropout_rate=0.5, batch_normalization=True, **_):
        super(NNBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.num_layers = num_hidden_layers

        self.input_layer = torch.nn.Sequential(torch.nn.Linear(input_dim, num_neurons), ACTIVATION)
        self.middle_layers = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(num_neurons, num_neurons), ACTIVATION) for _ in range(num_hidden_layers)])
        self.output_layer = torch.nn.Sequential(torch.nn.Linear(num_neurons, output_dim))

        if batch_normalization:
            self.input_layer.append(torch.nn.BatchNorm1d(num_neurons))
            for i in range(len(self.middle_layers)): self.middle_layers[i].append(torch.nn.BatchNorm1d(num_neurons))
        
        if dropout:
            self.input_layer.append(torch.nn.Dropout(dropout_rate))
            for i in range(len(self.middle_layers)): self.middle_layers[i].append(torch.nn.Dropout(dropout_rate))


    def forward(self, x):
        h = x.view(-1, self.input_dim)
        h = self.input_layer(h)
        for layer in self.middle_layers:h = layer(h)
        h = self.output_layer(h)
        return h

    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class ConvBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_neurons=50, num_hidden_layers=2, **_):
        super(ConvBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.num_layers = num_hidden_layers

        self.input_layer = torch.nn.Conv1d(input_dim, num_neurons, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle_layers = torch.nn.ModuleList([
                                                torch.nn.Sequential(torch.nn.Conv1d(num_neurons, num_neurons, kernel_size=3, padding=1), 
                                                torch.nn.MaxPool2d(kernel_size=2, stride=2), 
                                                torch.nn.Dropout(0.5), 
                                                torch.nn.BatchNorm2d(num_neurons)) for _ in range(num_hidden_layers)
                                                ])
        self.output_layer = torch.nn.Conv1d(num_neurons, output_dim, kernel_size=3, padding=1)

        # setup the non-linearity
        self.act = ACTIVATION

    def forward(self, x):
        h = x.view(-1, self.input_dim)
        h = self.act(self.input_layer(h))
        
        for layer in self.middle_layers:
            h = self.act(layer(h))

        h = self.output_layer(h)
        return h

    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ParameterizerNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, conv=False, dist_params=["mu"], num_hidden_layers=2, num_neurons=50, dropout=True, dropout_rate=0.5, batch_normalization=True, **_):
        super(ParameterizerNN, self).__init__()
        self.dist_params = dist_params
        self.block_dict = torch.nn.ModuleDict()

        if conv:
            self.block_dict["input"] = ConvBlock(input_dim, num_neurons, num_neurons=num_neurons, num_hidden_layers=num_hidden_layers, dropout=False, dropout_rate=dropout_rate, batch_normalization=False)
        else:
            self.block_dict["input"] = NNBlock(input_dim, num_neurons, num_neurons=num_neurons, num_hidden_layers=num_hidden_layers, dropout=False, dropout_rate=dropout_rate, batch_normalization=False)
        
        self.block_dict["input"].output_layer.append(ACTIVATION)
        # if batch_normalization: self.block_dict["input"].output_layer.append(torch.nn.BatchNorm1d(num_neurons))
        # if dropout: self.block_dict["input"].output_layer.append(torch.nn.Dropout(dropout_rate))

        for param in dist_params:
            self.block_dict[param] = NNBlock(num_neurons, output_dim, num_neurons=num_neurons, num_hidden_layers=0, dropout=True, dropout_rate=dropout_rate, batch_normalization=True)

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
    def __init__(self, input_dim, output_dim, learn_sigma=True, num_hidden_layers=2, num_neurons=50, dropout=True, dropout_rate=0.5, batch_normalization=True, **_):
        super(GaussianNN, self).__init__()

        self.learn_sigma = learn_sigma

        if learn_sigma: dist_params = ["mu", "sigma"]
        else: dist_params = ["mu"]

        self.parameterizer = ParameterizerNN(input_dim, output_dim, dist_params=dist_params, num_hidden_layers=num_hidden_layers, num_neurons=num_neurons, dropout=dropout, dropout_rate=dropout_rate, batch_normalization=batch_normalization)

    def _num_parameters(self):
        return self.parameterizer._num_parameters()
    
    def forward(self, inputs):
        param_dict = self.parameterizer(inputs)
        if self.learn_sigma: param_dict["sigma"] = to_sigma(param_dict["sigma"])
        else: param_dict["sigma"] = torch.ones_like(param_dict["mu"])
        return param_dict
    
    def rsample(self, param_dict=None, num_samples=1, **_):
        return param_dict["mu"] + param_dict["sigma"] * torch.randn((num_samples, *param_dict["mu"].shape))

    def sample(self, param_dict=None, num_samples=1, **_):
        return self.rsample(param_dict=param_dict, num_samples=num_samples)
    
    def log_likelihood(self, targets, param_dict=None):
        return log_prob("Normal", param_dict, targets)
    
    def kl_divergence(self, param_dict=None, prior_params={"mu":0.0, "sigma":1.0}):
        return kl_divergence("Normal", param_dict, prior_params)

class LogitNormalNN(GaussianNN):
    def __init__(self, input_dim, output_dim, conv=False, num_hidden_layers=2, num_neurons=50, dropout=True, dropout_rate=0.5, batch_normalization=True, **_):
        super(LogitNormalNN, self).__init__(input_dim=input_dim, output_dim=output_dim, num_hidden_layers=num_hidden_layers, num_neurons=num_neurons, learn_sigma=True, dropout=True, dropout_rate=0.5, batch_normalization=True)

    def _num_parameters(self): return super().parameterizer._num_parameters()
    
    def sample(self, param_dict=None, num_samples=1, **_): return super().sample(param_dict=param_dict, num_samples=num_samples).sigmoid()

    def rsample(self, param_dict=None, num_samples=1, **_): return super().rsample(param_dict=param_dict, num_samples=num_samples).sigmoid()
    
    def log_likelihood(self, targets, param_dict=None): return super().log_likelihood(targets.logit(), param_dict)

class DirichletNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, conv=False, num_hidden_layers=2, num_neurons=50, dropout=True, dropout_rate=0.5, batch_normalization=True, **_):
        super(DirichletNN, self).__init__()

        dist_params = ["alpha"]

        self.parameterizer = ParameterizerNN(input_dim, output_dim, dist_params=dist_params, num_hidden_layers=num_hidden_layers, num_neurons=num_neurons, dropout=dropout, dropout_rate=dropout_rate, batch_normalization=batch_normalization)

    def _num_parameters(self): return self.parameterizer._num_parameters()
    
    def forward(self, inputs, constrain=True):
        total_count_per_word = inputs.sum(dim=-1, keepdim=True)
        param_dict = self.parameterizer(inputs/total_count_per_word)
        if constrain: param_dict["alpha"] = to_alpha(param_dict["alpha"])
        return param_dict
    
    def sample(self, param_dict=None, num_samples=1, **_):
        return torch.distributions.Dirichlet(param_dict["alpha"]).sample((num_samples,))

    def rsample(self, param_dict=None, num_samples=1, **_):
        return torch.distributions.Dirichlet(param_dict["alpha"]).rsample((num_samples,))
    
    def log_likelihood(self, targets, param_dict=None):
        return log_prob("Dirichlet", param_dict, targets)
    
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
        return log_prob("Bernoulli", param_dict, targets)
    
    def kl_divergence(self, param_dict=None, prior_params={"pi":0.5}):
        return kl_divergence("Bernoulli", param_dict, prior_params)

def get_distribution_model(dist_type, **kwargs):
    if dist_type.lower() in ["gaussian", "gauss", "normal", "n", "g"]: return GaussianNN(**kwargs)
    elif dist_type.lower() in ["dirichlet", "dir", "d"]: return DirichletNN(**kwargs)
    elif dist_type.lower() in ["logitnormal", "ln"]: return LogitNormalNN(**kwargs)
    elif dist_type.lower() in ["bernoulli", "bern", "b"]: return BernoulliNN(**kwargs)
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