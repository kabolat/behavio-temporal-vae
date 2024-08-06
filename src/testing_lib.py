import torch
import numpy as np

@torch.no_grad()
def mass_reconstruction(model, x_test, conditions_test, num_mc_samples=1, device="cpu"):
    model.to(device)
    model.prior_params = {key: value.to(device) for key, value in model.prior_params.items()}

    x_rec_, z_rec_ = model.reconstruct(inputs=x_test.to(device), conditions=conditions_test.to(device))
    z_rec = {"params": {k: v.to("cpu") for k, v in z_rec_["params"].items()}}
    z_rec["samples"] = z_rec_["samples"].to("cpu").repeat_interleave(num_mc_samples, dim=0)
    x_rec = {"params": {k: v.to("cpu").repeat_interleave(num_mc_samples, dim=0) for k, v in x_rec_["params"].items()}}

    for i in range(1,num_mc_samples):
        z = model.encoder.rsample(z_rec_["params"], num_samples=1, **model.model_kwargs)[0]
        likelihood_params_dict = model.decoder(torch.cat((z,conditions_test.to(device)),dim=-1))

        z_rec["samples"][i] = z.to("cpu")
        del z

        for param in likelihood_params_dict.keys(): x_rec["params"][param][i] = likelihood_params_dict[param].to("cpu")
        del likelihood_params_dict
        if torch.cuda.is_available() and device!="cpu": torch.cuda.empty_cache()

    del x_rec_, z_rec_

    model.to("cpu")
    model.prior_params = {key: value.to("cpu") for key, value in model.prior_params.items()}

    return x_rec, z_rec


@torch.no_grad()
def mass_imputation(model, conditions_test, num_mc_samples_prior=1, num_mc_samples_likelihood=1, device="cpu"):
    model.to(device)
    model.prior_params = {key: value.to(device) for key, value in model.prior_params.items()}

    x_imp_ = model.sample(conditions_test.to(device))
    x_imp = {"params": {k: v.to("cpu").repeat_interleave(num_mc_samples_prior,dim=0)*0.0 for k, v in x_imp_["params"].items()}}
    x_imp["samples"] = x_imp_["samples"].to("cpu").repeat_interleave(num_mc_samples_prior,dim=1).repeat_interleave(num_mc_samples_likelihood,dim=0)*0.0
    del x_imp_

    for i in range(num_mc_samples_prior):
        x_imp_ = model.sample(conditions_test.to(device))
        for param in x_imp_["params"].keys(): x_imp["params"][param][i] = x_imp_["params"][param].to("cpu")
        x_imp["samples"][0,i] = x_imp_["samples"].to("cpu")
        for j in range(1, num_mc_samples_likelihood):
            x_imp["samples"][j,i] = model.decoder.sample(x_imp_["params"], **model.model_kwargs).to("cpu")

        del x_imp_
        if torch.cuda.is_available() and device!="cpu": torch.cuda.empty_cache()
    
    model.to("cpu")
    model.prior_params = {key: value.to("cpu") for key, value in model.prior_params.items()}

    return x_imp

@torch.no_grad()
def mass_denormalization(model, x_imp, nonzero_mean, nonzero_std, zero_id, shift, log_space, deviation=1, device="cpu"):
    x_imp_denorm = {"samples": x_imp["samples"]*0.0, "mean": x_imp["params"]["mu"]*0.0, "upper": x_imp["params"]["mu"]*0.0, "lower": x_imp["params"]["mu"]*0.0}
    X_sigma = model.decoder.get_marginal_sigmas(x_imp["params"])
    
    for i in range(x_imp["samples"].shape[1]):
        x_samples = x_imp["samples"][:,i].to(device)
        x_mean = x_imp["params"]["mu"][i].to(device)
        x_sigma = X_sigma[i].to(device)

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
            if typ=="samples": x_imp_denorm[typ][:,i] = x.cpu()
            else: x_imp_denorm[typ][i] = x.cpu()
            del x, x_log
    if torch.cuda.is_available(): torch.cuda.empty_cache()    
    return x_imp_denorm


@torch.no_grad()
def get_probabilistic_metrics(model, x_test, x_rec, z_rec, aggregate=True, device="cpu"):
    rlls = torch.zeros(x_rec["params"]["mu"].shape[:2])
    model.to(device)
    for i in range(x_rec["params"]["mu"].shape[0]):
        rlls[i] = model.decoder.log_likelihood(x_test.to(device), {k: v[i].to(device) for k, v in x_rec["params"].items()}).cpu()
    model.to("cpu")
    # rlls = model.decoder.log_likelihood(x_test, x_rec["params"])
    posterior_loglikelihood = model.encoder.log_likelihood(z_rec["samples"], z_rec["params"])
    prior_loglikelihood = model.encoder.log_likelihood(z_rec["samples"], model.prior_params)
    
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
    
def get_perplexity(per_user_loglikelihoods, num_samples):
    return np.exp(-np.sum([loglikelihood.sum() for loglikelihood in per_user_loglikelihoods]) / np.sum(num_samples))