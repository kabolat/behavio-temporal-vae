import torch
import json
from .submodels import *
from .utils import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class NeuralLLNA(torch.nn.Module):
    def __init__(self, 
                input_dim=None,
                num_topics=10,
                prior_param=None,
                prodlda = False,
                decoder_temperature = 1.0,
                conv=False,
                num_neurons=50,
                num_hidden_layers=2,
                dropout=True,
                dropout_rate=0.5,
                batch_normalization=True, **_):
        super(NeuralLLNA, self).__init__()
        #region Take the arguments
        kwargs = dict(locals())
        for key in ["self","__class__","input_dim"]: kwargs.pop(key)
        self.model_kwargs = kwargs.copy()
        for key in kwargs: setattr(self,key,kwargs[key])
        #endregion

        self.input_dim = input_dim

        self.encoder = get_distribution_model("logitnormal", input_dim=self.input_dim, output_dim=self.num_topics, **kwargs)
        self.decoder = BetaDecoder(input_dim=self.num_topics, output_dim=self.input_dim, temperature=decoder_temperature, **kwargs)
        if prior_param is not None:
            self.prior_params = {}
            for key in prior_param: self.prior_params[key] = torch.tensor([prior_param[key]]*self.num_topics, dtype=torch.float32)
        else:
            self.prior_params = get_prior_params("logitnormal", self.num_topics)

        # self.num_parameters = self.encoder._num_parameters()+self.decoder._num_parameters()

    def forward(self, inputs, mc_samples=1, prior_params=None):
        posterior_params_dict = self.encoder(inputs)
        theta = self.encoder.rsample(posterior_params_dict, num_samples=mc_samples, **self.model_kwargs)

        likelihood_params = self.decoder(theta).view(mc_samples,inputs.shape[0],self.input_dim)
        
        return {"params":likelihood_params}, {"params":posterior_params_dict, "samples": theta}
    
    def reconstruct(self, inputs, mc_samples=1):
        with torch.no_grad():
            x_dict, theta_dict = self.forward(inputs, mc_samples)
            return x_dict, theta_dict
    
    def reconstruction_loglikelihood(self, x, likelihood_params):
        return (x*likelihood_params.log()).sum(dim=-1).mean(dim=0)
    
    def kl_divergence(self, posterior_params, prior_params=None):
        return self.encoder.kl_divergence(posterior_params, prior_params=prior_params).sum(dim=1)

    def loss(self, x, likelihood_params, posterior_params, prior_params=None):
        rll = self.reconstruction_loglikelihood(x, likelihood_params).mean(dim=0)
        kl = self.kl_divergence(posterior_params,prior_params=prior_params).mean(dim=0)
        loss = -(rll-self.train_kwargs["beta"]*kl)
        return {"loss":loss, "elbo": rll-kl, "rll": rll, "kl": kl}
    
    def train_core(self, inputs, optim, **_):
        x_dict, theta_dict = self.forward(inputs, mc_samples=self.train_kwargs["mc_samples"],prior_params=self.prior_params)
        optim.zero_grad()
        loss = self.loss(inputs, x_dict["params"], theta_dict["params"], prior_params=self.prior_params)
        loss["loss"].backward()
        optim.step()
        return loss
    
    def train_verbose(self, pbar, itx, loss_dict):
        pbar.write(f"Iteration: {itx} -- ELBO={loss_dict['elbo'].item():.2e} / RLL={loss_dict['rll'].item():.2e} / KL={loss_dict['kl'].item():.2e}")
    
    def fit(self, 
            dataloader,
            lr = 1e-3,
            beta = 1,
            mc_samples = 1,
            learn_prior = False,
            epochs = 1000,
            verbose_freq = 100,
            tensorboard = True,
            tqdm_func = tqdm,
            **_):
        
        self.train(True)
        
        #region Take the arguments
        kwargs = dict(locals())
        for key in ["self","dataloader","tqdm_func"]: kwargs.pop(key)
        try:
            self.train_kwargs.update(kwargs)
        except:
            self.train_kwargs = kwargs
        #endregion

        if learn_prior:
            for key in self.prior_params: self.prior_params[key] = torch.nn.Parameter(self.prior_params[key])

        # region Initializations
        self.train_kwargs["batch_size"] = dataloader.batch_sampler.batch_size

        writer = SummaryWriter()
        log_dir = writer.log_dir
        with open(log_dir+'/model_args.json','w') as f: json.dump(self.model_kwargs,f,indent=4)
        with open(log_dir+'/train_args.json','w') as f: json.dump(self.train_kwargs,f,indent=4)
        if tensorboard: writer.file_writer.event_writer._logdir += "/tensorboard" 

        total_itx_per_epoch = ((dataloader.dataset.__len__())//dataloader.batch_sampler.batch_size + (dataloader.drop_last==False))
        pbar_epx, pbar_itx = tqdm_func(total=epochs, desc="Epoch"), tqdm_func(total=total_itx_per_epoch, desc="Iteration in Epoch")

        if learn_prior: optim = torch.optim.Adam([{'params':self.encoder.parameters()}, {'params':self.decoder.parameters()}] + [{'params':param for param in self.prior_params.values()}], lr=lr, betas=(0.99, 0.999))
        else: optim = torch.optim.Adam([{'params':self.encoder.parameters()}, {'params':self.decoder.parameters()}], lr=lr, betas=(0.99, 0.999))
        #endregion

        epx, itx = 0, 0
        for _ in range(epochs):
            epx += 1
            pbar_epx.update(1)
            pbar_itx.reset()

            for inputs in dataloader:
                itx += 1
                pbar_itx.update(1)
                loss = self.train_core(inputs, optim)
                # region Logging
                if itx%verbose_freq==0: 
                    self.train_verbose(pbar_itx, itx, loss)
                    if tensorboard: 
                        writer.add_scalars('Loss/ELBO', {'train':loss['elbo']}, itx)
                        writer.add_scalars('Loss/RLL', {'train':loss['rll']}, itx)
                        writer.add_scalars('Loss/KL', {'train':loss['kl']}, itx)
                #endregion

class NeuralLDA(NeuralLLNA):
    def __init__(self, 
                input_dim=None,
                num_topics=10,
                prior_param=None,
                prodlda = False,
                decoder_temperature = 1.0,
                conv=False,
                num_neurons=50,
                num_hidden_layers=2,
                dropout=True,
                dropout_rate=0.5,
                batch_normalization=True, **_):
        
        #region Take the arguments
        kwargs = dict(locals())
        for key in ["self","__class__", "input_dim"]: kwargs.pop(key)
        #endregion
        if prior_param is None:
            self.prior_params = from_alpha(get_prior_params("dirichlet", self.num_topics))
        else:
            kwargs["prior_param"]["alpha"] = from_alpha(torch.tensor(kwargs["prior_param"]["alpha"])).item()

        super(NeuralLDA, self).__init__(input_dim=input_dim, **kwargs)

        self.input_dim = input_dim
        self.num_topics = num_topics

        self.encoder = get_distribution_model("dirichlet", input_dim=self.input_dim, output_dim=self.num_topics, **self.model_kwargs)
        self.decoder = BetaDecoder(input_dim=self.num_topics, output_dim=self.input_dim, temperature=decoder_temperature, **self.model_kwargs)

    
    def train_core(self, inputs, optim, **_):
        prior_params_constrained = {"alpha":to_alpha(self.prior_params["alpha"])*1.0}
        x_dict, theta_dict = self.forward(inputs, mc_samples=self.train_kwargs["mc_samples"],prior_params=prior_params_constrained)
        optim.zero_grad()
        loss = self.loss(inputs, x_dict["params"], theta_dict["params"], prior_params=prior_params_constrained)
        loss["loss"].backward()
        optim.step()
        return loss
    
    def get_prior_params(self):
        return to_alpha(self.prior_params["alpha"])


class InductiveLDA(NeuralLDA):
    def __init__(self, 
                input_dim=None,
                num_topics=10,
                prior_param=None,
                prodlda = False,
                decoder_temperature = 1.0,
                encoder_temperature = 1.0,
                conv=False,
                num_neurons=50,
                num_hidden_layers=2,
                dropout=True,
                dropout_rate=0.5,
                batch_normalization=True):
        #region Take the arguments
        kwargs = dict(locals())
        for key in ["self","__class__", "input_dim"]: kwargs.pop(key)
        #endregion
        super(InductiveLDA, self).__init__(input_dim=input_dim, **kwargs)
        self.encoder_temperature = encoder_temperature

    def forward(self, inputs, mc_samples=1, prior_params=None):
        if prior_params is None: prior_params = {"alpha": to_alpha(self.prior_params["alpha"])}

        posterior_params_dict = self.encoder(inputs, constrain=False)

        # Beta = self.decoder.get_beta()

        zeta = (posterior_params_dict["alpha"]/self.encoder_temperature).softmax(dim=-1)

        # phi = Beta*zeta.unsqueeze(-1)
        # phi = phi/phi.sum(dim=1,keepdim=True)
        # posterior_params_dict["alpha"] = prior_params["alpha"] + (phi*inputs.unsqueeze(1)).sum(-1)
        # posterior_params_dict["alpha"] = prior_params["alpha"] + torch.matmul(phi,inputs.unsqueeze(-1)).squeeze()
        posterior_params_dict["alpha"] = prior_params["alpha"] + zeta*inputs.sum(dim=1).unsqueeze(1)

        theta = self.encoder.rsample(posterior_params_dict, num_samples=mc_samples, **self.model_kwargs)

        likelihood_params = self.decoder(theta).view(mc_samples,inputs.shape[0],self.input_dim)
        
        return {"params":likelihood_params}, {"params":posterior_params_dict, "samples": theta}

