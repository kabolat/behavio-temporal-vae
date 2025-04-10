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
                distribution_dict = {},
                **_):
        super(VAE, self).__init__()

        #region Take the arguments
        kwargs = dict(locals())
        for key in ["self","__class__","input_dim"]: kwargs.pop(key)
        self.model_kwargs = kwargs
        for key in kwargs: setattr(self,key,kwargs[key])
        #endregion

        self.input_dim = input_dim
        self.encoder = get_distribution_model(input_dim=self.input_dim, output_dim=self.latent_dim, **distribution_dict["posterior"])
        self.decoder = get_distribution_model(input_dim=self.latent_dim, output_dim=self.input_dim, **distribution_dict["likelihood"])

        self.prior_params = get_prior_params(distribution_dict["posterior"]["dist_type"], self.latent_dim)

        self.num_parameters = self.encoder._num_parameters() + self.decoder._num_parameters()
    
    def forward(self, inputs, num_mc_samples=1):
        posterior_params_dict = self.encoder(inputs)
        z = self.encoder.rsample(posterior_params_dict, num_samples=num_mc_samples, **self.model_kwargs)
        likelihood_params_dict = self.decoder(z)
        for param in likelihood_params_dict: likelihood_params_dict[param] = likelihood_params_dict[param].view(num_mc_samples,inputs.shape[0],self.input_dim)
        return {"params":likelihood_params_dict}, {"params":posterior_params_dict, "samples": z}

    @torch.no_grad()
    def sample(self, num_samples_prior=1, num_samples_likelihood=1):
        z = self.encoder.sample(param_dict=self.prior_params, num_samples=num_samples_prior).unsqueeze(1)
        param_dict = self.decoder(z)
        samples = self.decoder.sample(param_dict, num_samples=num_samples_likelihood)
        return {"params":param_dict, "samples": samples}
    
    @torch.no_grad()
    def reconstruct(self, inputs, num_mc_samples=1):
        x_dict, z_dict = self.forward(inputs=inputs, num_mc_samples=num_mc_samples)
        return x_dict, z_dict
    
    @torch.no_grad()
    def loglikelihood(self, inputs, num_mc_samples=1):
        x_dict, z_dict = self.forward(inputs=inputs, num_mc_samples=num_mc_samples)
        posterior_loglikelihood = self.encoder.log_likelihood(z_dict["samples"], z_dict["params"])
        likelihood_loglikelihood = self.decoder.log_likelihood(inputs[None,:], x_dict["params"])
        prior_loglikelihood = self.encoder.log_likelihood(z_dict["samples"], self.prior_params)
        return (posterior_loglikelihood.sum(-1) + likelihood_loglikelihood.sum(-1) - prior_loglikelihood.sum(-1)).mean(dim=0)
    
    def reconstruction_loglikelihood(self, x, likelihood_params):
        return self.decoder.log_likelihood(x, likelihood_params)
    
    def kl_divergence(self, posterior_params, prior_params=None):
        if prior_params is None: prior_params = self.prior_params
        return self.encoder.kl_divergence(posterior_params, prior_params=prior_params).sum(dim=-1)
    
    def loss(self, x, likelihood_params, posterior_params, beta=1.0, prior_params=None):
        rll = self.reconstruction_loglikelihood(x, likelihood_params).mean()
        kl = self.kl_divergence(posterior_params,prior_params=prior_params).mean()
        loss = -(rll-beta*kl)
        return {"loss":loss, "elbo": rll-kl, "rll": rll, "kl": kl}
    
    def train_core(self, inputs, optim, **_):
        x_dict, z_dict = self.forward(inputs, num_mc_samples=self.train_kwargs["num_mc_samples"])
        optim.zero_grad(set_to_none=True)
        loss = self.loss(inputs, x_dict["params"], z_dict["params"], beta=self.train_kwargs["beta"], prior_params=self.prior_params)
        loss["loss"].backward()
        if self.train_kwargs["gradient_clipping"]: torch.nn.utils.clip_grad_norm_(self.parameters(), **self.train_kwargs["gradient_clipping_kwargs"])
        optim.step()
        return loss
    
    def validate(self, valloader, num_mc_samples=1, device="cpu"):
        self.eval()
        val_loss = {"elbo":0.0, "rll":0.0, "kl":0.0}
        with torch.no_grad():
            for inputs in valloader:
                x_dict, z_dict = self.forward(self.move_to_device(inputs, device=device), num_mc_samples=num_mc_samples)
                loss = self.loss(inputs, x_dict["params"], z_dict["params"], beta=1.0, prior_params=self.prior_params)
                for key in val_loss: val_loss[key] += loss[key].item()*inputs.shape[0]
        for key in val_loss: val_loss[key] /= valloader.dataset.__len__()
        return val_loss

    def train_verbose(self, itx, loss_dict, pbar=None):
        if pbar is not None:
            pbar.write(f"Iteration: {itx} -- ELBO={loss_dict['elbo'].item():.4e} / RLL={loss_dict['rll'].item():.4e} / KL={loss_dict['kl'].item():.4e}")
        else:
            print(f"Iteration: {itx} -- ELBO={loss_dict['elbo'].item():.4e} / RLL={loss_dict['rll'].item():.4e} / KL={loss_dict['kl'].item():.4e}")
    
    def get_optimizer(self, lr=1e-3, weight_decay=1e-4):
        return torch.optim.Adam([{'params':self.encoder.parameters()}, {'params':self.decoder.parameters()}], lr=lr, weight_decay=weight_decay)
    
    def move_to_device(self, inputs, device="cpu"):
        return inputs.to(device)

    def save(self, save_path=None, model_name="trained_model", device="cpu"):
        if save_path is None: save_path = self.log_dir
        model_path = os.path.join(save_path, model_name + '.pt')
        self.eval()
        self.to("cpu")
        self.prior_params = {key: value.to("cpu") for key, value in self.prior_params.items()}
        torch.save(self.state_dict(), model_path)
        self.to(device)
        self.prior_params = {key: value.to(self.train_kwargs["device"]) for key, value in self.prior_params.items()}
    
    def load(self, load_path=None, model_name="trained_model"):
        if load_path is None: load_path = self.log_dir
        model_path = os.path.join(load_path, model_name + '.pt')
        self.load_state_dict(torch.load(model_path, weights_only=True))
        self.to("cpu")
        self.prior_params = {k: v.to("cpu") for k, v in self.prior_params.items()}
        self.eval()

    def fit(self, 
            trainloader,
            valloader = None,
            lr=1e-3,
            weight_decay=1e-4,
            gradient_clipping = False,
            gradient_clipping_kwargs = {"max_norm":1.0},
            lr_scheduling = False,
            lr_scheduling_kwargs = {"threshold":0.1, "factor":0.2, "patience":5, "min_lr":1e-5},
            beta=1.0,
            num_mc_samples=1,
            epochs=1000,
            verbose_freq=100,
            tensorboard=True,
            tqdm_func=tqdm,
            validation_freq=200,
            validation_mc_samples=1,
            device = "cpu",
            earlystopping = False,
            earlystopping_kwargs = {"patience":5, "delta":0.1},
            save_epoch_freq = 5,
            writer = None,
            **_):
        
        #region Take the arguments
        kwargs = dict(locals())
        for key in ["self","trainloader","valloader","tqdm_func", "writer"]: kwargs.pop(key)
        try: self.train_kwargs.update(kwargs)
        except: self.train_kwargs = kwargs

        flattened_model_kwargs, flattened_train_kwargs = {}, {}
        for key, value in self.model_kwargs.items():
            if key == "_":
                for sub_key, sub_value in value.items(): flattened_model_kwargs[sub_key] = sub_value
            else: flattened_model_kwargs[key] = value
        for key, value in self.train_kwargs.items():
            if key == "_":
                for sub_key, sub_value in value.items(): flattened_train_kwargs[sub_key] = sub_value
            else: flattened_train_kwargs[key] = value
        #endregion

        #region Tensorboard
        if writer is None: writer = SummaryWriter()
        log_dir = writer.log_dir
        with open(os.path.join(log_dir,'model_kwargs.json'),'w') as f: json.dump(flattened_model_kwargs,f,indent=4)
        with open(os.path.join(log_dir,'train_kwargs.json'),'w') as f: json.dump(flattened_train_kwargs,f,indent=4)
        if tensorboard: writer.file_writer.event_writer._logdir = os.path.join(writer.file_writer.event_writer._logdir, "tensorboard")
        self.log_dir = log_dir        

        if tqdm_func is not None:
            total_itx_per_epoch = ((trainloader.dataset.__len__())//trainloader.batch_sampler.batch_size + (trainloader.drop_last==False))
            pbar_epx, pbar_itx = tqdm_func(total=epochs, desc="Epoch", dynamic_ncols=True), tqdm_func(total=total_itx_per_epoch, desc="Iteration in Epoch", dynamic_ncols=True)
        #endregion

        optim = self.get_optimizer(lr=lr, weight_decay=weight_decay)
        if lr_scheduling: scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", threshold_mode="abs", **lr_scheduling_kwargs)

        self.to(device)
        self.prior_params = {key: value.to(self.train_kwargs["device"]) for key, value in self.prior_params.items()}

        if earlystopping: earlystopper = EarlyStopping(**earlystopping_kwargs)
        
        self.train()
        epx, itx = 0, 0
        for _ in range(epochs):
            epx += 1
            if tqdm_func is not None: 
                pbar_epx.update(1)
                pbar_itx.reset()

            for inputs in trainloader:
                itx += 1
                if tqdm_func is not None: pbar_itx.update(1)

                loss = self.train_core(self.move_to_device(inputs, device=self.train_kwargs["device"]), optim)
                
                ## region Validation
                if itx%validation_freq==0 and itx>0 and valloader is not None:
                    val_loss = self.validate(valloader, num_mc_samples=validation_mc_samples, device=self.train_kwargs["device"])
                    if tqdm_func is not None: pbar_itx.write(f"Validation -- ELBO={val_loss['elbo']:.4e} / RLL={val_loss['rll']:.4e} / KL={val_loss['kl']:.4e}")
                    else: print(f"Validation -- ELBO={val_loss['elbo']:.4e} / RLL={val_loss['rll']:.4e} / KL={val_loss['kl']:.4e}")
                    if lr_scheduling:
                        last_lr = scheduler.get_last_lr()[0]
                        scheduler.step(val_loss["elbo"])
                        if last_lr != scheduler.get_last_lr()[0]: 
                            if tqdm_func is not None: pbar_itx.write(f"Learning Rate Changed: {last_lr} -> {scheduler.get_last_lr()[0]}")
                            else: print(f"Learning Rate Changed: {last_lr} -> {scheduler.get_last_lr()[0]}")
                    if earlystopping: earlystopper(val_loss["elbo"], pbar=pbar_itx)
                    if tensorboard:
                        writer.add_scalars('Loss/ELBO', {'val':val_loss['elbo']}, itx)
                        writer.add_scalars('Loss/RLL', {'val':val_loss['rll']}, itx)
                        writer.add_scalars('Loss/KL', {'val':val_loss['kl']}, itx)
                    del val_loss
                    self.train()
                #endregion
                # region Logging
                if itx%verbose_freq==0: 
                    if tqdm_func is not None: self.train_verbose(itx, loss, pbar=pbar_itx)
                    else: self.train_verbose(itx, loss)
                    if tensorboard: 
                        writer.add_scalars('Loss/ELBO', {'train':loss['elbo']}, itx)
                        writer.add_scalars('Loss/RLL', {'train':loss['rll']}, itx)
                        writer.add_scalars('Loss/KL', {'train':loss['kl']}, itx)
                #endregion
                del loss
            if save_epoch_freq is not None and epx%save_epoch_freq==0: 
                self.save(device=device)
                if tqdm_func is not None: pbar_epx.write(f"Model Saved at Epoch {epx} to {self.log_dir}")
                else: print(f"Model Saved at Epoch {epx} to {self.log_dir}")
            if earlystopping and earlystopper.early_stop: break
        self.to("cpu")
        self.prior_params = {key: value.to("cpu") for key, value in self.prior_params.items()}
        self.eval()

class CVAE(VAE):
    def __init__(self, 
                input_dim = None,
                conditioner = None,
                latent_dim = 10,
                distribution_dict = {},
                **_):
        
        kwargs = dict(locals())
        for key in ["self","__class__", "input_dim", "conditioner"]: kwargs.pop(key)
        self.model_kwargs = kwargs
        super(CVAE, self).__init__(input_dim=input_dim, **kwargs)

        self.condition_dim = conditioner.cond_dim

        self.encoder = get_distribution_model(input_dim=self.input_dim +self.condition_dim, output_dim=self.latent_dim, **distribution_dict["posterior"])
        self.decoder = get_distribution_model(input_dim=self.latent_dim+self.condition_dim, output_dim=self.input_dim, **distribution_dict["likelihood"])

        self.prior_params = get_prior_params(distribution_dict["posterior"]["dist_type"], self.latent_dim)

    def forward(self, inputs, conditions, num_mc_samples=1):
        if conditions.ndim==2:
            posterior_params_dict = self.encoder(torch.cat((inputs,conditions),dim=-1))
            z = self.encoder.rsample(posterior_params_dict, num_samples=num_mc_samples, **self.model_kwargs)
            likelihood_params_dict = self.decoder(torch.cat((z,conditions.unsqueeze(0).repeat_interleave(num_mc_samples,dim=0)),dim=2))
            for param in likelihood_params_dict: likelihood_params_dict[param] = likelihood_params_dict[param].view(num_mc_samples,inputs.shape[0],likelihood_params_dict[param].shape[-1])
            return {"params":likelihood_params_dict}, {"params":posterior_params_dict, "samples": z}
        elif conditions.ndim==3:
            if conditions.shape[1] == num_mc_samples:
                conditions = conditions.transpose(1,0)
                inputs = inputs.unsqueeze(0).repeat_interleave(num_mc_samples,dim=0)
                posterior_params_dict = self.encoder(torch.cat((inputs,conditions),dim=-1))
                for param in posterior_params_dict: posterior_params_dict[param] = posterior_params_dict[param].view(num_mc_samples,inputs.shape[1],posterior_params_dict[param].shape[-1])
                z = self.encoder.rsample(posterior_params_dict, num_samples=1, **self.model_kwargs).squeeze(0)
                likelihood_params_dict = self.decoder(torch.cat((z,conditions),dim=-1))
                for param in likelihood_params_dict: likelihood_params_dict[param] = likelihood_params_dict[param].view(num_mc_samples,inputs.shape[1],likelihood_params_dict[param].shape[-1])
                return {"params":likelihood_params_dict}, {"params":posterior_params_dict, "samples": z}
            else:
                raise ValueError("Condition samples must be equal to number of posterior MC samples.")
        else:
            raise ValueError("Invalid shape for conditions. Must be 2D or 3D tensor.")

    @torch.no_grad
    def sample(self, condition, num_samples_prior=1, num_samples_likelihood=1):
        if condition.ndim==2:
            condition = condition.unsqueeze(0).repeat_interleave(num_samples_prior,dim=0)
            z = self.encoder.sample(param_dict=self.prior_params, num_samples=num_samples_prior*condition.shape[1]).reshape(num_samples_prior,condition.shape[1],-1)
            param_dict = self.decoder(torch.cat((z,condition),dim=2))
            param_dict = {key: value.view(num_samples_prior,condition.shape[1],-1) for key, value in param_dict.items()}
            samples = self.decoder.sample(param_dict, num_samples=num_samples_likelihood)
            return {"params":param_dict, "samples": samples}
        elif condition.ndim==3:
            if condition.shape[1] == num_samples_prior:
                condition = condition.transpose(1,0)
                z = self.encoder.sample(param_dict=self.prior_params, num_samples=num_samples_prior*condition.shape[1]).reshape(num_samples_prior,condition.shape[1],-1)
                param_dict = self.decoder(torch.cat((z,condition),dim=-1))
                param_dict = {key: value.view(num_samples_prior,condition.shape[1],-1) for key, value in param_dict.items()}
                samples = self.decoder.sample(param_dict, num_samples=num_samples_likelihood)
                return {"params":param_dict, "samples": samples}
            else:
                raise ValueError("Condition samples must be equal to number of prior samples.")
        else:
            raise ValueError("Invalid shape for conditions. Must be 2D or 3D tensor.")
    
    @torch.no_grad
    def reconstruct(self, inputs, conditions, num_mc_samples=1):
        x_dict, z_dict = self.forward(inputs=inputs, conditions=conditions, num_mc_samples=num_mc_samples)
        return x_dict, z_dict
    
    @torch.no_grad
    def loglikelihood(self, inputs, conditions, num_mc_samples=1):
        x_dict, z_dict = self.forward(inputs=inputs, conditions=conditions, num_mc_samples=num_mc_samples)
        posterior_loglikelihood = self.encoder.log_likelihood(z_dict["samples"], z_dict["params"])
        likelihood_loglikelihood = self.decoder.log_likelihood(inputs[None,:], x_dict["params"])
        prior_loglikelihood = self.encoder.log_likelihood(z_dict["samples"], self.prior_params)
        return -torch.log(torch.tensor(num_mc_samples)) + torch.logsumexp(-posterior_loglikelihood + likelihood_loglikelihood + prior_loglikelihood).mean(dim=0)

    def train_core(self, inputs, optim, **_):
        inputs, conditions = inputs
        x_dict, z_dict = self.forward(inputs, conditions, num_mc_samples=self.train_kwargs["num_mc_samples"])
        optim.zero_grad(set_to_none=True)
        loss = self.loss(inputs, x_dict["params"], z_dict["params"], beta=self.train_kwargs["beta"], prior_params=self.prior_params)
        loss["loss"].backward()
        if self.train_kwargs["gradient_clipping"]: torch.nn.utils.clip_grad_norm_(self.parameters(), **self.train_kwargs["gradient_clipping_kwargs"])
        optim.step()
        return loss
    
    @torch.no_grad
    def validate(self, valloader, num_mc_samples=1, beta=1.0, device="cpu"):
        self.eval()
        val_loss = {"elbo":0.0, "rll":0.0, "kl":0.0}
        for inputs in valloader:
            inputs, conditions = self.move_to_device(inputs, device=device)
            x_dict, z_dict = self.forward(inputs, conditions, num_mc_samples=num_mc_samples)
            loss = self.loss(inputs, x_dict["params"], z_dict["params"], beta=beta, prior_params=self.prior_params)
            for key in val_loss: val_loss[key] += loss[key].item()*inputs.shape[0]
        for key in val_loss: val_loss[key] /= valloader.dataset.__len__()
        return val_loss
    
    def move_to_device(self, inputs, device="cpu"): return [x.to(device) for x in inputs]