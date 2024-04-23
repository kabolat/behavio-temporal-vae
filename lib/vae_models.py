import torch
import json
from .submodels import *
from .utils import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
torch.autograd.set_detect_anomaly(True)


class VAE(torch.nn.Module):
    def __init__(self,
                input_dim = None,
                latent_dim = 10,
                posterior_dist = "normal",
                likelihood_dist = "normal",
                learn_decoder_sigma = True, 
                num_neurons = 50,
                num_hidden_layers = 2,
                dropout = True,
                dropout_rate = 0.5,
                batch_normalization = True,
                **_):
        super(VAE, self).__init__()

        #region Take the arguments
        kwargs = dict(locals())
        for key in ["self","__class__","input_dim"]: kwargs.pop(key)
        self.model_kwargs = kwargs
        for key in kwargs: setattr(self,key,kwargs[key])
        #endregion

        self.input_dim = input_dim
        self.encoder = get_distribution_model(self.posterior_dist,  input_dim=self.input_dim,  output_dim=self.latent_dim, learn_sigma=True, **kwargs)
        self.decoder = get_distribution_model(self.likelihood_dist, input_dim=self.latent_dim, output_dim=self.input_dim,  learn_sigma=self.learn_decoder_sigma, **kwargs)
        self.prior_params = get_prior_params(self.posterior_dist, self.latent_dim)

        self.num_parameters = self.encoder._num_parameters() + self.decoder._num_parameters()
    
    def forward(self, inputs, num_mc_samples=1):
        posterior_params_dict = self.encoder(inputs)
        z = self.encoder.rsample(posterior_params_dict, num_samples=num_mc_samples, **self.model_kwargs)
        likelihood_params_dict = self.decoder(z)
        for param in likelihood_params_dict: likelihood_params_dict[param] = likelihood_params_dict[param].view(num_mc_samples,inputs.shape[0],self.input_dim)
        return {"params":likelihood_params_dict}, {"params":posterior_params_dict, "samples": z}

    def sample(self, num_samples_prior=1, num_samples_likelihood=1):
        with torch.no_grad():
            z = self.encoder.sample(param_dict=self.prior_params, num_samples=num_samples_prior).unsqueeze(1)
            param_dict = self.decoder(z)
            samples = self.decoder.sample(param_dict, num_samples=num_samples_likelihood)
            return {"params":param_dict, "samples": samples}
    
    def reconstruct(self, inputs, num_mc_samples=1):
        with torch.no_grad():
            x_dict, z_dict = self.forward(inputs=inputs, num_mc_samples=num_mc_samples)
            return x_dict, z_dict
    
    def reconstruction_loglikelihood(self, x, likelihood_params):
        return self.decoder.log_likelihood(x, likelihood_params).sum(dim=2).mean(dim=0)
    
    def kl_divergence(self, posterior_params, prior_params=None):
        if prior_params is None: prior_params = self.prior_params
        return self.encoder.kl_divergence(posterior_params, prior_params=self.prior_params).sum(dim=1)
    
    def loss(self, x, likelihood_params, posterior_params, beta=1.0, prior_params=None):
        rll = self.reconstruction_loglikelihood(x, likelihood_params).mean(dim=0)
        kl = self.kl_divergence(posterior_params,prior_params=prior_params).mean(dim=0)
        loss = -(rll-beta*kl)
        return {"loss":loss, "elbo": rll-kl, "rll": rll, "kl": kl}
    
    def train_core(self, inputs, optim, **_):
        x_dict, z_dict = self.forward(inputs, num_mc_samples=self.train_kwargs["num_mc_samples"])
        optim.zero_grad()
        loss = self.loss(inputs, x_dict["params"], z_dict["params"], beta=self.train_kwargs["beta"], prior_params=self.prior_params)
        loss["loss"].backward()
        optim.step()
        return loss
    
    def validate(self, valloader, num_mc_samples=100):
        self.eval()
        val_loss = {"elbo":0.0, "rll":0.0, "kl":0.0}
        with torch.no_grad():
            for inputs in valloader:
                x_dict, z_dict = self.forward(inputs, num_mc_samples=num_mc_samples)
                loss = self.loss(inputs, x_dict["params"], z_dict["params"], beta=1.0, prior_params=self.prior_params)
                for key in val_loss: val_loss[key] += loss[key].item()*inputs.shape[0]
        for key in val_loss: val_loss[key] /= valloader.dataset.__len__()
        return val_loss

    def train_verbose(self, pbar, itx, loss_dict):
        pbar.write(f"Iteration: {itx} -- ELBO={loss_dict['elbo'].item():.2e} / RLL={loss_dict['rll'].item():.2e} / KL={loss_dict['kl'].item():.2e}")
    
    def get_optimizer(self, lr=1e-3):
        return torch.optim.Adam([{'params':self.encoder.parameters()}, {'params':self.decoder.parameters()}], lr=lr)

    def fit(  self, 
            trainloader,
            valloader = None,
            lr=1e-3,
            beta=1.0,
            num_mc_samples=1,
            epochs=1000,
            verbose_freq=100,
            tensorboard=True,
            tqdm_func=tqdm,
            validation_freq=200,
            **_):
        
        #region Take the arguments
        kwargs = dict(locals())
        for key in ["self","trainloader","valloader","tqdm_func"]: kwargs.pop(key)
        try: self.train_kwargs.update(kwargs)
        except: self.train_kwargs = kwargs
        #endregion

        flattened_model_kwargs, flattened_train_kwargs = {}, {}
        for key, value in self.model_kwargs.items():
            if isinstance(value, dict): 
                for sub_key, sub_value in value.items(): flattened_model_kwargs[sub_key] = sub_value
            else: flattened_model_kwargs[key] = value
        for key, value in self.train_kwargs.items():
            if isinstance(value, dict): 
                for sub_key, sub_value in value.items(): flattened_train_kwargs[sub_key] = sub_value
            else: flattened_train_kwargs[key] = value

        writer = SummaryWriter()
        log_dir = writer.log_dir
        with open(log_dir+'/model_args.json','w') as f: json.dump(flattened_model_kwargs,f,indent=4)
        with open(log_dir+'/train_args.json','w') as f: json.dump(flattened_train_kwargs,f,indent=4)
        if tensorboard: writer.file_writer.event_writer._logdir += "/tensorboard" 
        self.log_dir = log_dir

        total_itx_per_epoch = ((trainloader.dataset.__len__())//trainloader.batch_sampler.batch_size + (trainloader.drop_last==False))
        pbar_epx, pbar_itx = tqdm_func(total=epochs, desc="Epoch"), tqdm_func(total=total_itx_per_epoch, desc="Iteration in Epoch")

        optim = self.get_optimizer(lr=lr)
        #endregion

        self.train()
        epx, itx = 0, 0
        for _ in range(epochs):
            epx += 1
            pbar_epx.update(1)
            pbar_itx.reset()

            for inputs in trainloader:
                itx += 1
                pbar_itx.update(1)
                loss = self.train_core(inputs, optim)
                
                ## region Validation
                if itx%validation_freq==0 and itx>0 and valloader is not None:
                    val_loss = self.validate(valloader, num_mc_samples=100)
                    print(f"Validation -- ELBO={val_loss['elbo']:.2e} / RLL={val_loss['rll']:.2e} / KL={val_loss['kl']:.2e}")
                    if tensorboard:
                        writer.add_scalars('Loss/ELBO', {'val':val_loss['elbo']}, itx)
                        writer.add_scalars('Loss/RLL', {'val':val_loss['rll']}, itx)
                        writer.add_scalars('Loss/KL', {'val':val_loss['kl']}, itx)
                    self.train()
                #endregion
                # region Logging
                if itx%verbose_freq==0: 
                    self.train_verbose(pbar_itx, itx, loss)
                    if tensorboard: 
                        writer.add_scalars('Loss/ELBO', {'train':loss['elbo']}, itx)
                        writer.add_scalars('Loss/RLL', {'train':loss['rll']}, itx)
                        writer.add_scalars('Loss/KL', {'train':loss['kl']}, itx)
                #endregion
        self.eval()


class CVAE(VAE):
    def __init__(self, 
                input_dim = None,
                conditioner = None,
                latent_dim = 10,
                posterior_dist = "normal",
                likelihood_dist = "normal",
                learn_decoder_sigma = True,
                num_neurons = 50,
                num_hidden_layers = 2,
                dropout = True,
                dropout_rate = 0.5,
                batch_normalization = True,
                **_):
        
        kwargs = dict(locals())
        for key in ["self","__class__", "input_dim", "conditioner"]: kwargs.pop(key)
        self.model_kwargs = kwargs
        super(CVAE, self).__init__(input_dim=input_dim, **kwargs)

        self.condition_dim = conditioner.cond_dim

        self.encoder = get_distribution_model(self.posterior_dist,  input_dim=self.input_dim +self.condition_dim, output_dim=self.latent_dim, learn_sigma=True, **_)
        self.decoder = get_distribution_model(self.likelihood_dist, input_dim=self.latent_dim+self.condition_dim, output_dim=self.input_dim,  learn_sigma=self.learn_decoder_sigma, **_)

        self.prior_params = get_prior_params(self.posterior_dist, self.latent_dim)

    def forward(self, inputs, conditions, num_mc_samples=1):
        posterior_params_dict = self.encoder(torch.cat((inputs,conditions),dim=1))
        z = self.encoder.rsample(posterior_params_dict, num_samples=num_mc_samples, **self.model_kwargs)
        likelihood_params_dict = self.decoder(torch.cat((z,conditions.unsqueeze(0).repeat_interleave(num_mc_samples,dim=0)),dim=2))
        for param in likelihood_params_dict: likelihood_params_dict[param] = likelihood_params_dict[param].view(num_mc_samples,inputs.shape[0],self.input_dim)
        return {"params":likelihood_params_dict}, {"params":posterior_params_dict, "samples": z}

    def sample(self, condition, num_samples_prior=1, num_samples_likelihood=1):
        with torch.no_grad():
            condenc = condition.unsqueeze(0).repeat_interleave(num_samples_prior,dim=0)
            z = self.encoder.sample(param_dict=self.prior_params, num_samples=num_samples_prior).unsqueeze(1)
            param_dict = self.decoder(torch.cat((z,condenc),dim=2))
            samples = self.decoder.sample(param_dict, num_samples=num_samples_likelihood)
            return {"params":param_dict, "samples": samples}
    
    def reconstruct(self, inputs, conditions, num_mc_samples=1):
        with torch.no_grad():
            x_dict, z_dict = self.forward(inputs=inputs, conditions=conditions, num_mc_samples=num_mc_samples)
            return x_dict, z_dict
    
    def train_core(self, inputs, optim, **_):
        inputs, conditions = inputs
        x_dict, z_dict = self.forward(inputs, conditions, num_mc_samples=self.train_kwargs["num_mc_samples"])
        optim.zero_grad()
        loss = self.loss(inputs, x_dict["params"], z_dict["params"], beta=self.train_kwargs["beta"], prior_params=self.prior_params)
        loss["loss"].backward()
        optim.step()
        return loss
    
    def validate(self, valloader, num_mc_samples=100):
        self.eval()
        val_loss = {"elbo":0.0, "rll":0.0, "kl":0.0}
        with torch.no_grad():
            for inputs, conditions in valloader:
                x_dict, z_dict = self.forward(inputs, conditions, num_mc_samples=num_mc_samples)
                loss = self.loss(inputs, x_dict["params"], z_dict["params"], beta=1.0, prior_params=self.prior_params)
                for key in val_loss: val_loss[key] += loss[key].item()*inputs.shape[0]
        for key in val_loss: val_loss[key] /= valloader.dataset.__len__()
        return val_loss
    
    def get_optimizer(self, lr=1e-3):
        return torch.optim.Adam([{'params':self.encoder.parameters()}, {'params':self.decoder.parameters()}], lr=lr)