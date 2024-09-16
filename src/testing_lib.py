import torch
import numpy as np

@torch.no_grad()
def mass_loglikelihood(model, x, conditions, num_mc_samples=1, batch_size=10000, mc_sample_batch_size=100, device="cpu"):

    if num_mc_samples<=mc_sample_batch_size:
        mc_sample_batch_size = num_mc_samples
        print(f"Setting the batch size for MC samples to {mc_sample_batch_size}.")
    
    data_size = x.shape[0]
    x_rec_, z_rec_ = model.reconstruct(inputs=x[[0]], conditions=conditions[[0]])
    z_rec = {"params": {k: v.repeat_interleave(data_size, dim=0)*0.0 for k, v in z_rec_["params"].items()}}
    z_rec["samples"] = z_rec_["samples"].repeat_interleave(num_mc_samples, dim=0).repeat_interleave(data_size, dim=1)*0.0
    x_rec = {"params": {k: v.repeat_interleave(num_mc_samples, dim=0).repeat_interleave(data_size, dim=1)*0.0 for k, v in x_rec_["params"].items()}}

    rlls = torch.zeros(x_rec["params"]["mu"].shape[:2])

    model.to(device)
    model.prior_params = {key: value.to(device) for key, value in model.prior_params.items()}

    dataset = torch.utils.data.TensorDataset(x, conditions)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=4)

    num_mc_epochs = int(np.ceil(num_mc_samples/mc_sample_batch_size))

    for k, (x_batch, conditions_batch) in enumerate(dataloader):

        print(f"Batch {k+1}/{len(dataloader)}")

        x_device = x_batch.to(device)
        
        x_rec__, z_rec__ = model.reconstruct(inputs=x_device, conditions=conditions_batch.to(device), num_mc_samples=1)

        z_rec_ = {"params": z_rec__["params"].copy()}
        z_rec_["samples"] = z_rec__["samples"].repeat_interleave(num_mc_samples, dim=0)*0.0
        x_rec_ = {"params": {k: v.repeat_interleave(num_mc_samples, dim=0)*0.0 for k, v in x_rec__["params"].items()}}

        del x_rec__, z_rec__

        for i in range(num_mc_epochs):
            z = model.encoder.rsample(z_rec_["params"], num_samples=mc_sample_batch_size, **model.model_kwargs)
            x_rec_batch_ = model.decoder(torch.cat((z,conditions_batch.unsqueeze(0).repeat_interleave(mc_sample_batch_size,dim=0).to(device)),dim=-1))
            x_rec_batch = {}

            if (i+1)*mc_sample_batch_size < num_mc_samples:

                for param in x_rec_["params"].keys(): 
                    x_rec_batch[param] = x_rec_batch_[param].view(mc_sample_batch_size, -1, x_rec_["params"][param].shape[-1])
                    x_rec_["params"][param][i*mc_sample_batch_size:(i+1)*mc_sample_batch_size] = x_rec_batch[param]
                
                rlls[i*mc_sample_batch_size:(i+1)*mc_sample_batch_size, k*batch_size:(k+1)*batch_size] = model.decoder.log_likelihood(x_device, x_rec_batch).cpu()

                z_rec_["samples"][i*mc_sample_batch_size:(i+1)*mc_sample_batch_size] = z

            else:

                for param in x_rec_["params"].keys(): 
                    x_rec_batch[param] = x_rec_batch_[param].view(mc_sample_batch_size, -1, x_rec_["params"][param].shape[-1])
                    x_rec_["params"][param][i*mc_sample_batch_size:] = x_rec_batch[param]
                
                rlls[i*mc_sample_batch_size:, k*batch_size:(k+1)*batch_size] = model.decoder.log_likelihood(x_device, x_rec_batch).cpu()

                z_rec_["samples"][i*mc_sample_batch_size:] = z

            del z, x_rec_batch, x_rec_batch_
        
        if (k+1)*batch_size < data_size:
            for param in z_rec_["params"].keys(): z_rec["params"][param][k*batch_size:(k+1)*batch_size] = z_rec_["params"][param].to("cpu")
            for param in x_rec_["params"].keys(): x_rec["params"][param][:,k*batch_size:(k+1)*batch_size] = x_rec_["params"][param].to("cpu")
            z_rec["samples"][:,k*batch_size:(k+1)*batch_size] = z_rec_["samples"].to("cpu")
        else:
            for param in z_rec_["params"].keys(): z_rec["params"][param][k*batch_size:] = z_rec_["params"][param].to("cpu")
            for param in x_rec_["params"].keys(): x_rec["params"][param][:,k*batch_size:] = x_rec_["params"][param].to("cpu")
            z_rec["samples"][:,k*batch_size:] = z_rec_["samples"].to("cpu")

        del x_rec_, z_rec_
        if torch.cuda.is_available() and device!="cpu": torch.cuda.empty_cache()

    model.to("cpu")
    model.prior_params = {key: value.to("cpu") for key, value in model.prior_params.items()}

    posterior_loglikelihood = model.encoder.log_likelihood(z_rec["samples"], z_rec["params"])
    prior_loglikelihood = model.encoder.log_likelihood(z_rec["samples"], model.prior_params)

    loglikelihood = -torch.log(torch.tensor(rlls.shape[0])) + torch.logsumexp(-posterior_loglikelihood + rlls + prior_loglikelihood, dim=0)

    return loglikelihood.numpy()

@torch.no_grad()
def mass_reconstruction(model, x_test, conditions_test, num_mc_samples=1, batch_size=10000, mc_sample_batch_size=100, device="cpu"):

    data_size = x_test.shape[0]
    x_rec_, z_rec_ = model.reconstruct(inputs=x_test[[0]], conditions=conditions_test[[0]])
    z_rec = {"params": {k: v.repeat_interleave(data_size, dim=0)*0.0 for k, v in z_rec_["params"].items()}}
    z_rec["samples"] = z_rec_["samples"].repeat_interleave(num_mc_samples, dim=0).repeat_interleave(data_size, dim=1)*0.0
    x_rec = {"params": {k: v.repeat_interleave(num_mc_samples, dim=0).repeat_interleave(data_size, dim=1)*0.0 for k, v in x_rec_["params"].items()}}

    model.to(device)
    model.prior_params = {key: value.to(device) for key, value in model.prior_params.items()}

    dataset = torch.utils.data.TensorDataset(x_test, conditions_test)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=4)

    num_mc_epochs = int(np.ceil(num_mc_samples/mc_sample_batch_size))

    for k, (x_batch, conditions_batch) in enumerate(dataloader):
        
        x_rec__, z_rec__ = model.reconstruct(inputs=x_batch.to(device), conditions=conditions_batch.to(device), num_mc_samples=1)

        z_rec_ = {"params": z_rec__["params"].copy()}
        z_rec_["samples"] = z_rec__["samples"].repeat_interleave(num_mc_samples, dim=0)*0.0
        x_rec_ = {"params": {k: v.repeat_interleave(num_mc_samples, dim=0)*0.0 for k, v in x_rec__["params"].items()}}

        del x_rec__, z_rec__

        for i in range(num_mc_epochs):
            if (i+1)*mc_sample_batch_size < num_mc_samples:
                z = model.encoder.rsample(z_rec_["params"], num_samples=mc_sample_batch_size, **model.model_kwargs)
                for param in x_rec_["params"].keys(): x_rec_["params"][param][i*mc_sample_batch_size:(i+1)*mc_sample_batch_size] = model.decoder(torch.cat((z,conditions_batch.unsqueeze(0).repeat_interleave(mc_sample_batch_size,dim=0).to(device)),dim=-1))[param].view(mc_sample_batch_size, -1, x_rec_["params"][param].shape[-1])
                z_rec_["samples"][i*mc_sample_batch_size:(i+1)*mc_sample_batch_size] = z
                del z
            else:
                z = model.encoder.rsample(z_rec_["params"], num_samples=num_mc_samples-i*mc_sample_batch_size, **model.model_kwargs)
                for param in x_rec_["params"].keys(): x_rec_["params"][param][i*mc_sample_batch_size:] = model.decoder(torch.cat((z,conditions_batch.unsqueeze(0).repeat_interleave(num_mc_samples-i*mc_sample_batch_size,dim=0).to(device)),dim=-1))[param].view(num_mc_samples-i*mc_sample_batch_size, -1, x_rec_["params"][param].shape[-1])
                z_rec_["samples"][i*mc_sample_batch_size:] = z
                del z

        if (k+1)*batch_size < data_size:
            for param in z_rec_["params"].keys(): z_rec["params"][param][k*batch_size:(k+1)*batch_size] = z_rec_["params"][param].to("cpu")
            for param in x_rec_["params"].keys(): x_rec["params"][param][:,k*batch_size:(k+1)*batch_size] = x_rec_["params"][param].to("cpu")
            z_rec["samples"][:,k*batch_size:(k+1)*batch_size] = z_rec_["samples"].to("cpu")
        else:
            for param in z_rec_["params"].keys(): z_rec["params"][param][k*batch_size:] = z_rec_["params"][param].to("cpu")
            for param in x_rec_["params"].keys(): x_rec["params"][param][:,k*batch_size:] = x_rec_["params"][param].to("cpu")
            z_rec["samples"][:,k*batch_size:] = z_rec_["samples"].to("cpu")

        del x_rec_, z_rec_
        if torch.cuda.is_available() and device!="cpu": torch.cuda.empty_cache()

        print(f"Batch {k+1}/{len(dataloader)}")

    model.to("cpu")
    model.prior_params = {key: value.to("cpu") for key, value in model.prior_params.items()}

    return x_rec, z_rec


@torch.no_grad()
def mass_imputation(model, conditions_test, num_mc_samples_prior=1, num_mc_samples_likelihood=1, batch_size=10000, mc_sample_batch_size=100, device="cpu"):

    data_size = conditions_test.shape[0]
    x_imp_ = model.sample(conditions_test[[0]])
    x_imp = {"params": {k: v.repeat_interleave(num_mc_samples_prior,dim=0).repeat_interleave(data_size, dim=1)*0.0 for k, v in x_imp_["params"].items()}}
    x_imp["samples"] = x_imp_["samples"].repeat_interleave(num_mc_samples_likelihood, dim=0).repeat_interleave(num_mc_samples_prior, dim=1).repeat_interleave(data_size, dim=2)*0.0

    model.to(device)
    model.prior_params = {key: value.to(device) for key, value in model.prior_params.items()}

    dataset = torch.utils.data.TensorDataset(conditions_test)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=4)

    num_mc_epochs = int(np.ceil(num_mc_samples_prior/mc_sample_batch_size))

    for k, (conditions_batch,) in enumerate(dataloader):
        x_imp__ = model.sample(conditions_batch.to(device), num_samples_prior=1, num_samples_likelihood=num_mc_samples_likelihood)

        x_imp_ = {"params": {k: v.repeat_interleave(num_mc_samples_prior, dim=0)*0.0 for k, v in x_imp__["params"].items()}}
        x_imp_["samples"] = x_imp__["samples"].repeat_interleave(num_mc_samples_prior, dim=1)*0.0

        del x_imp__

        for i in range(num_mc_epochs):
            if (i+1)*mc_sample_batch_size < num_mc_samples_prior:
                x_imp__ = model.sample(conditions_batch.to(device), num_samples_prior=mc_sample_batch_size, num_samples_likelihood=num_mc_samples_likelihood)
                for param in x_imp_["params"].keys(): x_imp_["params"][param][i*mc_sample_batch_size:(i+1)*mc_sample_batch_size] = x_imp__["params"][param]
                x_imp_["samples"][:,i*mc_sample_batch_size:(i+1)*mc_sample_batch_size] = x_imp__["samples"]
                del x_imp__
            else:
                x_imp__ = model.sample(conditions_batch.to(device), num_samples_prior=num_mc_samples_prior-i*mc_sample_batch_size, num_samples_likelihood=num_mc_samples_likelihood)
                for param in x_imp_["params"].keys(): x_imp_["params"][param][i*mc_sample_batch_size:] = x_imp__["params"][param]
                x_imp_["samples"][:,i*mc_sample_batch_size:] = x_imp__["samples"]
                del x_imp__

        if (k+1)*batch_size < data_size:
            for param in x_imp_["params"].keys(): x_imp["params"][param][:,k*batch_size:(k+1)*batch_size] = x_imp_["params"][param].to("cpu")
            x_imp["samples"][:,:,k*batch_size:(k+1)*batch_size] = x_imp_["samples"].to("cpu")

        del x_imp_
        if torch.cuda.is_available() and device!="cpu": torch.cuda.empty_cache()

        print(f"Batch {k+1}/{len(dataloader)}")

    model.to("cpu")
    model.prior_params = {key: value.to("cpu") for key, value in model.prior_params.items()}

    return x_imp

@torch.no_grad()
def mass_denormalization(model, x_imp, nonzero_mean, nonzero_std, zero_id, shift, log_space, deviation=None, batch_size=10000, device="cpu"):
    if deviation is None: 
        x_imp_denorm = {"samples": x_imp["samples"]*0.0, "mean": x_imp["params"]["mu"]*0.0}
        X_sigma = None
    else: 
        x_imp_denorm = {"samples": x_imp["samples"], "mean": x_imp["params"]["mu"], "upper": x_imp["params"]["mu"], "lower": x_imp["params"]["mu"]}
        X_sigma = model.decoder.get_marginal_sigmas(x_imp["params"])

    num_epochs = int(np.ceil(x_imp["samples"].shape[-2]/batch_size))

    for k in range(num_epochs):
        for i in range(x_imp["samples"].shape[1]):

            if (k+1)*batch_size<x_imp["samples"].shape[-2]:
                x_samples = x_imp["samples"][:,i,k*batch_size:(k+1)*batch_size].to(device)
                x_mean = x_imp["params"]["mu"][i,k*batch_size:(k+1)*batch_size].to(device)
                if X_sigma is not None: x_sigma = X_sigma[i,k*batch_size:(k+1)*batch_size].to(device)
            else:
                x_samples = x_imp["samples"][:,i,k*batch_size:].to(device)
                x_mean = x_imp["params"]["mu"][i,k*batch_size:].to(device)
                if X_sigma is not None: x_sigma = X_sigma[i,k*batch_size:].to(device)

            for typ in x_imp_denorm.keys():
                if typ=="samples": x = x_samples
                elif typ=="mean": x = x_mean
                elif typ=="upper": x = x_mean + deviation*x_sigma
                elif typ=="lower": x = x_mean - deviation*x_sigma
                else: raise ValueError("Unknown type.")

                is_zero = (x <= zero_id+1e-2)
                x[is_zero] = torch.nan
                if log_space: x_log = x
                else: x_log = torch.log(x)
                x_log = (x_log-torch.tensor(shift).to(device))*torch.tensor(nonzero_std).to(device) + torch.tensor(nonzero_mean).to(device)
                x = torch.exp(x_log)
                x[is_zero] = 0
                if typ=="samples": x_imp_denorm[typ][:,i,k*batch_size:(k+1)*batch_size] = x.cpu()
                else: x_imp_denorm[typ][i,k*batch_size:(k+1)*batch_size] = x.cpu()
                del x, x_log
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # print(f"Batch {k+1}/{num_epochs}")
    return x_imp_denorm


@torch.no_grad()
def get_probabilistic_metrics(model, x, x_rec, z_rec, aggregate=True, batch_size=10000, device="cpu"):
    rlls = torch.zeros(x_rec["params"]["mu"].shape[:2])
    model.to(device)
    x_device = x.to(device)
    for i in range(x_rec["params"]["mu"].shape[0]):
        x_rec_device = {k: v[i].to(device) for k, v in x_rec["params"].items()}
        rlls[i] = model.decoder.log_likelihood(x_device, x_rec_device).cpu() ##BUG: Decoder is passed for each parameter!
        del x_rec_device
        if torch.cuda.is_available() and device!="cpu": torch.cuda.empty_cache()

    posterior_loglikelihood = model.encoder.log_likelihood(z_rec["samples"], z_rec["params"])
    prior_loglikelihood = model.encoder.log_likelihood(z_rec["samples"], model.prior_params)
    model.to("cpu")
    
    loglikelihood = -torch.log(torch.tensor(rlls.shape[0])) + torch.logsumexp(-posterior_loglikelihood + rlls + prior_loglikelihood, dim=0)
       
    elbo = rlls.mean(dim=0) - model.kl_divergence(z_rec["params"], prior_params=model.prior_params)

    metrics = {}
    for metric_name in ["loglikelihood", "elbo"]:
        if metric_name=="elbo": metric = elbo
        else: metric = loglikelihood
        if aggregate:
            metrics[metric_name] = {}
            metrics[metric_name]["mean"] = metric.mean().item()
            metrics[metric_name]["std"] = metric.std().item()
            metrics[metric_name]["median"] = metric.median().item()
            metrics[metric_name]["quartile_lower"] = metric.quantile(0.25).item()
            metrics[metric_name]["quartile_upper"] = metric.quantile(0.75).item()
            metrics[metric_name]["min"] = metric.min().item()
            metrics[metric_name]["max"] = metric.max().item()
        else: metrics[metric_name] = metric.numpy()
    if torch.cuda.is_available() and device!="cpu": torch.cuda.empty_cache()   
    return metrics

@torch.no_grad()
def get_sample_metrics(x_test, x_imp, imputation_style="samples", aggregate=True):
    if imputation_style not in ["samples", "mean"]: raise ValueError("Unknown imputation style.")

    rmse = (((x_test - x_imp[imputation_style])**2).mean(dim=-1)**.5).mean(dim=0)
    if imputation_style=="samples": rmse = rmse.mean(dim=0)

    metrics = {}
    for metric_name in ["rmse"]:
        metric = rmse
        if aggregate:
            metrics[metric_name] = {}
            metrics[metric_name]["mean"] = metric.mean().item()
            metrics[metric_name]["std"] = metric.std().item()
            metrics[metric_name]["median"] = metric.median().item()
            metrics[metric_name]["quartile_lower"] = metric.quantile(0.25).item()
            metrics[metric_name]["quartile_upper"] = metric.quantile(0.75).item()
            metrics[metric_name]["min"] = metric.min().item()
            metrics[metric_name]["max"] = metric.max().item()
        else: metrics[metric_name] = metric.numpy()
    return metrics

@torch.no_grad()
def get_per_user_metrics(metrics, user_ids, num_users, summary=False):
    metrics_per_user = [metrics[np.where(user_ids==i)[0]] for i in range(num_users)]
    if not summary: return metrics_per_user
    else: return {"mean": np.array([np.nanmean(estimates) if len(estimates)>0 else np.nan for estimates in metrics_per_user]), 
                  "std": np.array([np.nanstd(estimates) if len(estimates)>0 else np.nan for estimates in metrics_per_user]), 
                  "median": np.array([np.nanmedian(estimates) if len(estimates)>0 else np.nan for estimates in metrics_per_user]), 
                  "quartile_lower": np.array([np.nanpercentile(estimates, 25) if len(estimates)>0 else np.nan for estimates in metrics_per_user]),
                  "quartile_upper": np.array([np.nanpercentile(estimates, 75) if len(estimates)>0 else np.nan for estimates in metrics_per_user]),
                  "min": np.array([np.nanmin(estimates) if len(estimates)>0 else np.nan for estimates in metrics_per_user]), 
                  "max": np.array([np.nanmax(estimates) if len(estimates)>0 else np.nan for estimates in metrics_per_user]), 
                  }