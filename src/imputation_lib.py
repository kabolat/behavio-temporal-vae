import torch
import numpy as np

@torch.no_grad()
def get_conditonal_params(model, param_dict, x_observed, observed_indices):
    mu = param_dict["mu"]
    Sigma = model.create_covariance_matrix(param_dict)
    
    unobserved_indices = torch.tensor([i for i in range(model.output_dim) if i not in observed_indices])
    
    mu_o = mu[:,observed_indices]
    mu_u = mu[:,unobserved_indices]

    Sigma_oo = Sigma[:,observed_indices][:,:, observed_indices]
    Sigma_ou = Sigma[:,observed_indices][:,:, unobserved_indices]
    Sigma_uu = Sigma[:,unobserved_indices][:,:, unobserved_indices]

    Sigma_oo_inv = torch.linalg.inv(Sigma_oo)
    conditional_mean = mu_u + ((x_observed - mu_o)[:,None,...] @ Sigma_oo_inv @ Sigma_ou).squeeze()
    conditional_cov = Sigma_uu - Sigma_ou.mT @ Sigma_oo_inv @ Sigma_ou
    return conditional_mean, conditional_cov

@torch.no_grad()
def sample_conditional(model, param_dict, x_observed, observed_indices, num_samples=1):

    conditional_mean, conditional_cov = get_conditonal_params(model, param_dict, x_observed, observed_indices)

    eps = torch.randn((num_samples, x_observed.shape[0], model.output_dim - len(observed_indices)), device=conditional_mean.device)
    L_cond = torch.linalg.cholesky(conditional_cov)
    cond_samples = conditional_mean + (L_cond @ eps.unsqueeze(-1)).squeeze(-1)

    return cond_samples, conditional_mean, conditional_cov

@torch.no_grad()
def pseudo_gibbs(model, x, missing_mask, conditioner, condition_set, num_samples=10, num_iter=100, device="cpu"):

    x_samples, z_samples, conditions_list = list(), list(), list()
    
    observed_indices = torch.where((~missing_mask)[0])[0] ## WARNING: It works only if the missing parts in the batch are all the same!

    for itx in range(num_iter):
        conditions = torch.tensor(conditioner.transform({k: v for k, v in condition_set.items()})).float().to(device)
        x_rec, z_rec = model.forward(x, conditions)

        cond_sample, _, _ = sample_conditional(model.decoder, {k:v[0] for k,v in x_rec["params"].items()}, x[:,observed_indices], observed_indices)
        x[missing_mask] = cond_sample.flatten()

        x_samples.append(x.cpu().tolist())
        z_samples.append(z_rec["samples"][0].cpu().tolist())
        conditions_list.append(conditions.cpu().tolist())

        if itx>=num_samples: 
            x_samples.pop(0)
            z_samples.pop(0)
            conditions_list.pop(0)

    return np.array(x_samples), np.array(z_samples), np.array(conditions_list)

@torch.no_grad()
def metropolis_within_gibbs(model, x, missing_mask, conditioner, condition_set, z_init=None, condition_init=None, num_samples=10, num_iter=100, verbose_freq=1000, device="cpu"):
    
    x_samples, z_samples, conditions_list = list(), list(), list()

    observed_indices = torch.where((~missing_mask)[0])[0] ## WARNING: It works only if the missing parts in the batch are all the same!
    
    if condition_init is None: conditions = torch.tensor(conditioner.transform({k: v for k, v in condition_set.items()})).float().to(device)
    else: conditions = condition_init

    if z_init is None: z = model.encoder.sample(model.encoder(torch.cat((x, conditions), dim=-1)))
    else: z = z_init

    for itx in range(num_iter):

        if itx%verbose_freq == 0:
            num_accepted_in_epoch = 0
            print(f"Iteration: {itx}")

        z_params = model.encoder(torch.cat((x, conditions), dim=-1))

        x_params = model.decoder(torch.cat((z, conditions), dim=-1))

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

        cond_sample, _, _ = sample_conditional(model.decoder, x_params, x[:,observed_indices], observed_indices)
        
        x[missing_mask] = cond_sample.flatten()

        x_samples.append(x.cpu().tolist())
        z_samples.append(z.cpu().tolist())
        conditions_list.append(conditions.cpu().tolist())

        if itx >= num_samples: 
            x_samples.pop(0)
            z_samples.pop(0)
            conditions_list.pop(0)

        conditions = torch.tensor(conditioner.transform({k: v for k, v in condition_set.items()})).float().to(device)

        if (itx+1)%verbose_freq==0: print(f"Acceptance ratio: %{num_accepted_in_epoch/(verbose_freq*x.shape[0])*100}")
    return np.array(x_samples), np.array(z_samples), np.array(conditions_list)

@torch.no_grad()
def cvae_imputation(model, data, conditioner, condition_set, num_samples=10, num_iter=100, warmup_steps=10, verbose_freq=1000, device="cpu"):
    model.to(device)
    model.prior_params = {key: value.to(device) for key, value in model.prior_params.items()}

    x = data.copy()
    x = torch.tensor(x).float().to(device)
    missing_mask = torch.isnan(x)
    x[missing_mask] = 0.0

    if warmup_steps>0:
        print("Pseudo-Gibbs warm-up is starting...")
        x_samples, z_samples, condition_samples = pseudo_gibbs(model, x, missing_mask, conditioner, condition_set, num_samples=num_samples, num_iter=warmup_steps, device=device)
        print("Pseudo-Gibbs warm-up has ended.")
    else: x_samples, z_samples, condition_samples = x.cpu().numpy(), [None], [None]

    if num_iter > 0:
        print("Metropolis-within-Gibbs is starting...")
        x_samples, z_samples, condition_samples = metropolis_within_gibbs(model, torch.tensor(x_samples[0]).float().to(device), 
                                                                          missing_mask, 
                                                                          conditioner, 
                                                                          condition_set, 
                                                                          z_init=torch.tensor(z_samples[0]).float().to(device), 
                                                                          condition_init=torch.tensor(condition_samples[0]).float().to(device), 
                                                                          num_samples=num_samples, 
                                                                          num_iter=num_iter, 
                                                                          verbose_freq=verbose_freq, 
                                                                          device=device)
        print("Metropolis-within-Gibbs has ended.")

    model.to("cpu")
    model.prior_params = {key: value.to("cpu") for key, value in model.prior_params.items()}

    return x_samples, z_samples, condition_samples


