import torch
import numpy as np
from . import submodels
import time

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        wrapper.total_time += elapsed_time
        return result
    wrapper.total_time = 0  # Initialize total time for the function
    return wrapper

@timing_decorator
@torch.no_grad()
def get_conditional_params(model, param_dict, x_observed, observed_indices):
    unobserved_indices = torch.tensor([i for i in range(model.output_dim) if i not in observed_indices])

    if isinstance(model, submodels.DictionaryGaussian):
        mu = param_dict["mu"]
        Sigma = model.create_covariance_matrix(param_dict)
        
        mu_o = mu[...,observed_indices]
        mu_u = mu[...,unobserved_indices]

        Sigma_oo = Sigma[...,observed_indices,:][...,:, observed_indices]
        Sigma_uo = Sigma[...,unobserved_indices,:][...,:, observed_indices]
        Sigma_uu = Sigma[...,unobserved_indices,:][...,:, unobserved_indices]

        Sigma_oo_inv = torch.linalg.inv(Sigma_oo)
        Sigma_uo_oo_inv = Sigma_uo @ Sigma_oo_inv
        conditional_mean = mu_u + (Sigma_uo_oo_inv @ (x_observed - mu_o)[..., None])[...,0]
        conditional_cov = Sigma_uu - Sigma_uo_oo_inv @ Sigma_uo.mT
    elif isinstance(model, submodels.GaussianNN):
        conditional_mean = param_dict["mu"][...,unobserved_indices]
        conditional_cov = model.create_covariance_matrix({"mu": param_dict["mu"][...,observed_indices], "sigma": param_dict["sigma"][...,observed_indices]})
    else:
        raise NotImplementedError("This type of decoder is not implemented yet!")

    return conditional_mean, conditional_cov

@timing_decorator
@torch.no_grad()
def sample_conditional(model, param_dict, x_observed, observed_indices, num_samples=1):

    conditional_mean, conditional_cov = get_conditional_params(model, param_dict, x_observed, observed_indices)

    if isinstance(model, submodels.DictionaryGaussian): 
        L = torch.linalg.cholesky(conditional_cov).to(conditional_cov.device)
        cond_samples = conditional_mean + (L @ torch.randn_like(conditional_mean).to(conditional_mean.device)[...,None])[...,0]
    elif isinstance(model, submodels.GaussianNN):
        cond_samples = torch.randn_like(conditional_mean).to(conditional_mean.device)*torch.sqrt(torch.diagonal(conditional_cov,dim1=-2,dim2=-1)) + conditional_mean
    else: raise NotImplementedError("This type of decoder is not implemented yet!")

    return cond_samples, conditional_mean, conditional_cov

@timing_decorator
@torch.no_grad()
def pseudo_gibbs(model, x, missing_mask, conditioner, condition_set, num_samples=10, num_iter=100, verbose_freq=1000, device="cpu"):

    observed_indices = torch.where((~missing_mask)[0])[0] ## WARNING: It works only if the missing parts in the batch are all the same!

    conditions_slack = torch.zeros(1, conditioner.cond_dim).to(device)
    x_rec, z_rec = model.forward(x[[0]], conditions_slack)
    z_rec["samples"] = z_rec["samples"].expand(num_samples, x.shape[0], -1).clone()*0.0
    z_rec["params"] = {k:v[None,...].expand(num_samples, x.shape[0], -1).clone()*0.0 for k,v in z_rec["params"].items()}
    x_params = x_rec["params"]
    cond_x_samples, cond_x_mean, cond_x_cov = sample_conditional(model.decoder, x_params, x[[0]][:,observed_indices], observed_indices)
    cond_x_samples = cond_x_samples.expand(num_samples, x.shape[0], -1).clone()*0.0
    cond_x_mean = cond_x_mean.expand(num_samples, x.shape[0], -1).clone()*0.0
    cond_x_cov = cond_x_cov.expand(num_samples, x.shape[0], -1, -1).clone()*0.0
    x_rec["params"] = {k:v.expand(num_samples, x.shape[0], -1).clone()*0.0 for k,v in x_rec["params"].items()}

    conditions = torch.tensor(conditioner.transform(condition_set)).float().to(device)
    if conditions.ndim == 3: conditions = conditions.squeeze(0)

    if "dir" in conditioner.types and conditioner.transformers[conditioner.tags[conditioner.types.index("dir")]].transform_style == "sample": random_conditioning = True
    else: random_conditioning = False

    for itx in range(num_iter):
        if itx%verbose_freq == 0: print(f"Iteration: {itx}")

        x_rec_, z_rec_ = model.forward(x, conditions)

        cond_x_samples[itx%num_samples], cond_x_mean[itx%num_samples], cond_x_cov[itx%num_samples] = sample_conditional(model.decoder, {k:v[0] for k,v in x_rec_["params"].items()}, x[:,observed_indices], observed_indices)
        for k,v in x_rec_["params"].items(): x_rec["params"][k][itx%num_samples] = v[0].clone()
        for k,v in z_rec_["params"].items(): z_rec["params"][k][itx%num_samples] = v[0].clone()
        z_rec["samples"][itx%num_samples] = z_rec_["samples"][0].clone()

        x[missing_mask] = cond_x_samples[itx%num_samples].flatten()

        if random_conditioning:
            conditions = torch.tensor(conditioner.transform(condition_set)).float().to(device)
            if conditions.ndim == 3: conditions = conditions.squeeze(0)

    return cond_x_samples, cond_x_mean, cond_x_cov, x_rec, z_rec

@timing_decorator
@torch.no_grad()
def metropolis_within_gibbs(model, x, missing_mask, conditioner, condition_set, num_samples=10, num_iter=100, verbose_freq=1000, device="cpu"):
    
    observed_indices = torch.where((~missing_mask)[0])[0] ## WARNING: It works only if the missing parts in the batch are all the same!

    conditions = torch.tensor(conditioner.transform(condition_set)).float().to(device)
    if conditions.ndim == 3: conditions = conditions.squeeze(0)

    z = model.encoder.sample(model.encoder(torch.cat((x, conditions), dim=-1)))[0]

    x_params = model.decoder(torch.cat((z[[0]], conditions[[0]]), dim=-1))
    cond_x_samples, cond_x_mean, cond_x_cov = sample_conditional(model.decoder, x_params, x[[0]][:,observed_indices], observed_indices)
    cond_x_samples = cond_x_samples[None,...].expand(num_samples, x.shape[0], -1)*0.0
    cond_x_mean = cond_x_mean[None,...].expand(num_samples, x.shape[0], -1)*0.0
    cond_x_cov = cond_x_cov[None,...].expand(num_samples, x.shape[0], -1, -1)*0.0

    if "dir" in conditioner.types and conditioner.transformers[conditioner.tags[conditioner.types.index("dir")]].transform_style == "sample": random_conditioning = True
    else: random_conditioning = False

    for itx in range(num_iter):
        if itx%verbose_freq == 0:
            num_accepted_in_epoch = 0
            print(f"Iteration: {itx}")

        z_params = model.encoder(torch.cat((x, conditions), dim=-1))

        x_params = model.decoder(torch.cat((z, conditions), dim=-1))
        
        if random_conditioning:
            conditions = torch.tensor(conditioner.transform(condition_set)).float().to(device)
            if conditions.ndim == 3: conditions = conditions.squeeze(0)

        z_proposal = model.encoder.sample(model.encoder(torch.cat((x, conditions), dim=-1)))[0]

        x_params_proposal = model.decoder(torch.cat((z_proposal, conditions), dim=-1))

        loglikelihood_proposal = model.decoder.log_likelihood(x, param_dict=x_params_proposal) + model.encoder.log_likelihood(z_proposal, param_dict=model.prior_params) - model.encoder.log_likelihood(z_proposal, param_dict=z_params)

        loglikelihood = model.decoder.log_likelihood(x, param_dict=x_params) + model.encoder.log_likelihood(z, param_dict=model.prior_params) - model.encoder.log_likelihood(z, param_dict=z_params)

        log_ratio = loglikelihood_proposal - loglikelihood

        rho = torch.minimum(torch.tensor([1]), torch.exp(log_ratio.cpu()))

        eps = torch.rand((rho.shape[0],))
        accept_idx = torch.where(eps<rho)[0]
        z[accept_idx] = z_proposal[accept_idx]*1.0
        for k,v in x_params_proposal.items(): x_params[k][accept_idx] = v[accept_idx]*1.0
        num_accepted_in_epoch += accept_idx.shape[0]

        cond_x_samples[itx%num_samples], cond_x_mean[itx%num_samples], cond_x_cov[itx%num_samples] = sample_conditional(model.decoder, x_params, x[:,observed_indices], observed_indices)
        
        x[missing_mask] = cond_x_samples[itx%num_samples].flatten()

        if (itx+1)%verbose_freq==0: print(f"Acceptance ratio: %{num_accepted_in_epoch/(verbose_freq*x.shape[0])*100}")
    return cond_x_samples, cond_x_mean, cond_x_cov

@torch.no_grad()
def cvae_imputation(model, data, conditioner, condition_set, missing_data_init=None, num_samples=10, num_iter=100, warmup_steps=10, verbose_freq=1000, device="cpu"):
    model.to(device)
    model.prior_params = {key: value.to(device) for key, value in model.prior_params.items()}

    x = data.copy()
    x = torch.tensor(x).float().to(device)
    missing_mask = torch.isnan(x)
    if missing_data_init is None: x[missing_mask] = x[~missing_mask].mean()
    else: x[missing_mask] = missing_data_init

    if warmup_steps>0:
        print("Pseudo-Gibbs is starting...")
        cond_x_samples, cond_x_mean, cond_x_cov, x_rec, z_rec = pseudo_gibbs(model, x, missing_mask, conditioner, condition_set, num_samples=num_samples, num_iter=warmup_steps, verbose_freq=verbose_freq, device=device)
        if num_iter > 0: x[missing_mask] = cond_x_samples[warmup_steps%num_samples].flatten()
        print("Pseudo-Gibbs has ended.")
        print(f"Percentage time spent in get_conditional_params: {get_conditional_params.total_time/pseudo_gibbs.total_time*100:.2f}%")
        print(f"Percentage time spent in sample_conditional: {sample_conditional.total_time/pseudo_gibbs.total_time*100:.2f}%")
    else: cond_x_samples, cond_x_mean, cond_x_cov = x, [None], [None]

    if num_iter > 0:
        print("Metropolis-within-Gibbs is starting...")
        cond_x_samples, cond_x_mean, cond_x_cov = metropolis_within_gibbs(model, x, missing_mask, conditioner, condition_set, num_samples=num_samples, num_iter=num_iter, verbose_freq=verbose_freq, device=device)
        print("Metropolis-within-Gibbs has ended.")
        print(f"Percentage time spent in get_conditional_params: {get_conditional_params.total_time/metropolis_within_gibbs.total_time*100:.2f}%")
        print(f"Percentage time spent in sample_conditional: {sample_conditional.total_time/metropolis_within_gibbs.total_time*100:.2f}%")
        x_rec, z_rec = None, None
    
    model.to("cpu")
    model.prior_params = {key: value.to("cpu") for key, value in model.prior_params.items()}

    return cond_x_samples, cond_x_mean, cond_x_cov, x_rec, z_rec

@torch.no_grad()
def mass_cvae_imputation(model, data, conditioner, condition_set, batch_size=1000, num_samples=10, num_iter=100, warmup_steps=10, verbose_freq=1000, device="cpu"):
    model.to(device)
    model.prior_params = {key: value.to(device) for key, value in model.prior_params.items()}

    missing_idx = np.where(np.isnan(data[0]))

    cond_x_samples = np.zeros((num_samples, data.shape[0], len(missing_idx[0]))).astype(np.float32)
    cond_x_mean = np.zeros((num_samples, data.shape[0], len(missing_idx[0]))).astype(np.float32)
    cond_x_cov = np.zeros((num_samples, data.shape[0], len(missing_idx[0]), len(missing_idx[0]))).astype(np.float32)

    num_epochs = int(np.ceil(data.shape[0]/batch_size))

    for i in range(num_epochs):
        print(f"Batch {i+1}/{num_epochs}")
        if i<num_epochs-1:
            cond_x_samples[:,i*batch_size:(i+1)*batch_size], cond_x_mean[:,i*batch_size:(i+1)*batch_size], cond_x_cov[:,i*batch_size:(i+1)*batch_size] = cvae_imputation(model, data[i*batch_size:(i+1)*batch_size], conditioner, {k:v[i*batch_size:(i+1)*batch_size] for k,v in condition_set.items()}, num_samples=num_samples, num_iter=num_iter, warmup_steps=warmup_steps, verbose_freq=verbose_freq, device=device)
        else:
            cond_x_samples[:,i*batch_size:], cond_x_mean[:,i*batch_size:], cond_x_cov[:,i*batch_size:] = cvae_imputation(model, data[i*batch_size:], conditioner, {k:v[i*batch_size:] for k,v in condition_set.items()}, num_samples=num_samples, num_iter=num_iter, warmup_steps=warmup_steps, verbose_freq=verbose_freq, device=device)

    model.to("cpu")
    model.prior_params = {key: value.to("cpu") for key, value in model.prior_params.items()}

    return cond_x_samples, cond_x_mean, cond_x_cov

@torch.no_grad()
def mass_cvae_imputation_with_loglikelihood(model, data, target_data, conditioner, condition_set, missing_data_init=None, batch_size=1000, num_samples=10, num_iter=100, warmup_steps=10, verbose_freq=1000, device="cpu"):
    model.to(device)
    model.prior_params = {key: value.to(device) for key, value in model.prior_params.items()}

    loglikelihoods = np.ndarray(shape=(data.shape[0],), dtype=np.float32)
    cond_x_samples = np.zeros((num_samples, data.shape[0], target_data.shape[1])).astype(np.float32)

    num_epochs = int(np.ceil(data.shape[0]/batch_size))

    for i in range(num_epochs):
        print(f"Batch {i+1}/{num_epochs}")
        if i<num_epochs-1:
            cond_x_samples_, cond_x_mean, cond_x_cov, _, _ = cvae_imputation(model, data[i*batch_size:(i+1)*batch_size], conditioner, {k:v[i*batch_size:(i+1)*batch_size] for k,v in condition_set.items()}, missing_data_init=missing_data_init, num_samples=num_samples, num_iter=num_iter, warmup_steps=warmup_steps, verbose_freq=verbose_freq, device=device)
            conditional_cholesky_ = torch.linalg.cholesky(cond_x_cov.cpu())
            loglikelihoods_conditional = (torch.logsumexp(torch.distributions.MultivariateNormal(cond_x_mean.cpu(), scale_tril=conditional_cholesky_).log_prob(torch.tensor(target_data[i*batch_size:(i+1)*batch_size]).float()), dim=0) - np.log(num_samples))
            loglikelihoods[i*batch_size:(i+1)*batch_size] = loglikelihoods_conditional.detach().cpu().numpy()
            cond_x_samples[:,i*batch_size:(i+1)*batch_size] = cond_x_samples_.detach().cpu().numpy()
        else:
            cond_x_samples_, cond_x_mean, cond_x_cov, _, _ = cvae_imputation(model, data[i*batch_size:], conditioner, {k:v[i*batch_size:] for k,v in condition_set.items()},  missing_data_init=missing_data_init, num_samples=num_samples, num_iter=num_iter, warmup_steps=warmup_steps, verbose_freq=verbose_freq, device=device)
            conditional_cholesky_ = torch.linalg.cholesky(cond_x_cov.cpu())
            loglikelihoods_conditional = (torch.logsumexp(torch.distributions.MultivariateNormal(cond_x_mean.cpu(), scale_tril=conditional_cholesky_).log_prob(torch.tensor(target_data[i*batch_size:]).float()), dim=0) - np.log(num_samples))
            loglikelihoods[i*batch_size:] = loglikelihoods_conditional.detach().cpu().numpy()
            cond_x_samples[:,i*batch_size:] = cond_x_samples_.detach().cpu().numpy()
            
        del cond_x_samples_, cond_x_mean, cond_x_cov, conditional_cholesky_, loglikelihoods_conditional
        torch.cuda.empty_cache()

    model.to("cpu")
    model.prior_params = {key: value.to("cpu") for key, value in model.prior_params.items()}

    return loglikelihoods, cond_x_samples

