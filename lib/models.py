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
                prodlda = False,
                conv=False,
                num_neurons=50,
                num_hidden_layers=2,):
        super(NeuralLLNA, self).__init__()
        #region Take the arguments
        kwargs = dict(locals())
        for key in ["self","__class__","input_dim"]: kwargs.pop(key)
        self.model_kwargs = kwargs
        for key in kwargs: setattr(self,key,kwargs[key])
        #endregion

        self.input_dim = input_dim
        self.num_topics = num_topics

        self.encoder = get_distribution_model("logitnormal", input_dim=self.input_dim, output_dim=self.num_topics, **kwargs)
        self.decoder = BetaDecoder(input_dim=self.num_topics, output_dim=self.input_dim, **kwargs)
        self.prior_params = get_prior_params("logitnormal", self.num_topics)

        # self.num_parameters = self.encoder._num_parameters()+self.decoder._num_parameters()

    def forward(self, inputs, mc_samples=1):
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
        if prior_params is None: prior_params = self.prior_params
        return self.encoder.kl_divergence(posterior_params, prior_params=self.prior_params).sum(dim=1)

    def loss(self, x, likelihood_params, posterior_params, prior_params=None):
        rll = self.reconstruction_loglikelihood(x, likelihood_params).mean(dim=0)
        kl = self.kl_divergence(posterior_params,prior_params=prior_params).mean(dim=0)
        loss = -(rll-self.train_kwargs["beta"]*kl)
        return {"loss":loss, "elbo": rll-kl, "rll": rll, "kl": kl}
    
    def train_core(self, inputs, optim, **_):
        x_dict, theta_dict = self.forward(inputs, mc_samples=self.train_kwargs["mc_samples"])
        optim.zero_grad()
        loss = self.loss(inputs, x_dict["params"], theta_dict["params"], prior_params=self.prior_params)
        loss["loss"].backward()
        optim.step()
        return loss
    
    def train_verbose(self, pbar, itx, loss_dict):
        pbar.write(f"Iteration: {itx} -- ELBO={loss_dict['elbo'].item():.2e} / RLL={loss_dict['rll'].item():.2e} / KL={loss_dict['kl'].item():.2e}")
    
    def train(self, 
            dataloader,
            lr = 1e-3,
            beta = 1,
            mc_samples = 1,
            prior_params = None,
            learn_prior = False,
            epochs = 1000,
            verbose_freq = 100,
            tensorboard = True,
            tqdm_func = tqdm,
            **_):
        
        #region Take the arguments
        kwargs = dict(locals())
        for key in ["self","dataloader","tqdm_func"]: kwargs.pop(key)
        try:
            self.train_kwargs.update(kwargs)
        except:
            self.train_kwargs = kwargs
        #endregion
        
        if prior_params is not None:
            for key in prior_params: self.prior_params[key] = torch.tensor([prior_params[key]]*self.num_topics, dtype=torch.float32)

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

        if learn_prior: optim = torch.optim.Adam([{'params':self.encoder.parameters()}, {'params':self.decoder.parameters()}] + [{'params':param for param in self.prior_params.values()}], lr=lr)
        else: optim = torch.optim.Adam([{'params':self.encoder.parameters()}, {'params':self.decoder.parameters()}], lr=lr)
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
                prodlda = False,
                conv=False,
                num_neurons=50,
                num_hidden_layers=2,):
        super(NeuralLDA, self).__init__(input_dim=input_dim, num_topics=num_topics, prodlda=prodlda, conv=conv, num_neurons=num_neurons, num_hidden_layers=num_hidden_layers)

        self.encoder = get_distribution_model("dirichlet", input_dim=self.input_dim, output_dim=self.num_topics, **self.model_kwargs)
        self.prior_params = get_prior_params("dirichlet", self.num_topics)