import torch
import numpy as np
from . import datasets

@torch.no_grad()
def mass_loglikelihood(model, x, conditioner, condition_set, num_mc_samples=1, batch_size=10000, mc_sample_batch_size=100, device="cpu"):

    _, z_rec, rlls = mass_reconstruction(model, x, conditioner, condition_set, num_mc_samples=num_mc_samples, batch_size=batch_size, mc_sample_batch_size=mc_sample_batch_size, device=device)

    posterior_loglikelihood = model.encoder.log_likelihood(z_rec["samples"], z_rec["params"])
    prior_loglikelihood = model.encoder.log_likelihood(z_rec["samples"], model.prior_params)
    kl_divergence = model.kl_divergence(z_rec["params"], prior_params=model.prior_params)

    loglikelihood = -torch.log(torch.tensor(rlls.shape[0])) + torch.logsumexp(-posterior_loglikelihood + rlls + prior_loglikelihood, dim=0)

    return loglikelihood.numpy(), rlls.numpy(), kl_divergence.numpy()


@torch.no_grad()
def mass_reconstruction(model, x, conditioner, condition_set, num_mc_samples=1, batch_size=10000, mc_sample_batch_size=100, device="cpu"):
    if num_mc_samples<=mc_sample_batch_size:
        mc_sample_batch_size = num_mc_samples
        print(f"Setting the batch size for MC samples to {mc_sample_batch_size}.")
    
    data_size = x.shape[0]
    random_conditioning = "dir" in conditioner.types and conditioner.transformers[conditioner.tags[conditioner.types.index("dir")]].transform_style == "sample"
    x_rec_, z_rec_ = model.reconstruct(inputs=torch.zeros(1,model.input_dim), conditions=torch.zeros(1,conditioner.cond_dim))

    if random_conditioning: z_rec = {"params": {k: v[None,...].repeat_interleave(num_mc_samples, dim=0).repeat_interleave(data_size, dim=1)*0.0 for k, v in z_rec_["params"].items()}}
    else: z_rec = {"params": {k: v.repeat_interleave(data_size, dim=0)*0.0 for k, v in z_rec_["params"].items()}}

    z_rec["samples"] = z_rec_["samples"].repeat_interleave(num_mc_samples, dim=0).repeat_interleave(data_size, dim=1)*0.0
    x_rec = {"params": {k: v.repeat_interleave(num_mc_samples, dim=0).repeat_interleave(data_size, dim=1)*0.0 for k, v in x_rec_["params"].items()}}

    rlls = torch.zeros(x_rec["params"]["mu"].shape[:2])

    model.to(device)
    model.prior_params = {key: value.to(device) for key, value in model.prior_params.items()}

    dataset = datasets.ConditionedDataset(x, condition_set, conditioner, num_samples=num_mc_samples) ## if there is no sampling in any condition transformer, num_samples is not used
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=4)

    num_mc_epochs = int(np.ceil(num_mc_samples/mc_sample_batch_size))

    for k, (x_batch, conditions_batch) in enumerate(dataloader):

        print(f"Batch {k+1}/{len(dataloader)}")

        x_device = x_batch.to(device)

        if random_conditioning:
            for i in range(num_mc_epochs):
                if (i+1)*mc_sample_batch_size < num_mc_samples:
                    conditions = conditions_batch[:,i*mc_sample_batch_size:(i+1)*mc_sample_batch_size,:].to(device)
                    x_dict, z_dict = model.reconstruct(inputs=x_device, conditions=conditions, num_mc_samples=mc_sample_batch_size)

                    if (k+1)*batch_size < data_size:
                        for param in x_rec["params"].keys(): x_rec["params"][param][i*mc_sample_batch_size:(i+1)*mc_sample_batch_size,k*batch_size:(k+1)*batch_size] = x_dict["params"][param].cpu()
                        for param in z_rec["params"].keys(): z_rec["params"][param][i*mc_sample_batch_size:(i+1)*mc_sample_batch_size,k*batch_size:(k+1)*batch_size] = z_dict["params"][param].cpu()
                        z_rec["samples"][i*mc_sample_batch_size:(i+1)*mc_sample_batch_size,k*batch_size:(k+1)*batch_size] = z_dict["samples"].cpu()
                        rlls[i*mc_sample_batch_size:(i+1)*mc_sample_batch_size, k*batch_size:(k+1)*batch_size] = model.decoder.log_likelihood(x_device, x_dict["params"]).cpu()
                    else:
                        for param in x_rec["params"].keys(): x_rec["params"][param][i*mc_sample_batch_size:(i+1)*mc_sample_batch_size,k*batch_size:] = x_dict["params"][param].cpu()
                        for param in z_rec["params"].keys(): z_rec["params"][param][i*mc_sample_batch_size:(i+1)*mc_sample_batch_size,k*batch_size:] = z_dict["params"][param].cpu()
                        z_rec["samples"][i*mc_sample_batch_size:(i+1)*mc_sample_batch_size,k*batch_size:] = z_dict["samples"].cpu()
                        rlls[i*mc_sample_batch_size:(i+1)*mc_sample_batch_size, k*batch_size:] = model.decoder.log_likelihood(x_device, x_dict["params"]).cpu()
                else:
                    conditions = conditions_batch[:,i*mc_sample_batch_size:,:].to(device)
                    x_dict, z_dict = model.reconstruct(inputs=x_device, conditions=conditions, num_mc_samples=num_mc_samples-i*mc_sample_batch_size)
                    if (k+1)*batch_size < data_size:
                        for param in x_rec["params"].keys(): x_rec["params"][param][i*mc_sample_batch_size:,k*batch_size:(k+1)*batch_size] = x_dict["params"][param].cpu()
                        for param in z_rec["params"].keys(): z_rec["params"][param][i*mc_sample_batch_size:,k*batch_size:(k+1)*batch_size] = z_dict["params"][param].cpu()
                        z_rec["samples"][i*mc_sample_batch_size:,k*batch_size:(k+1)*batch_size] = z_dict["samples"].cpu()
                        rlls[i*mc_sample_batch_size:, k*batch_size:(k+1)*batch_size] = model.decoder.log_likelihood(x_device, x_dict["params"]).cpu()
                    else:
                        for param in x_rec["params"].keys(): x_rec["params"][param][i*mc_sample_batch_size:,k*batch_size:] = x_dict["params"][param].cpu()
                        for param in z_rec["params"].keys(): z_rec["params"][param][i*mc_sample_batch_size:,k*batch_size:] = z_dict["params"][param].cpu()
                        z_rec["samples"][i*mc_sample_batch_size:,k*batch_size:] = z_dict["samples"].cpu()
                        rlls[i*mc_sample_batch_size:, k*batch_size:] = model.decoder.log_likelihood(x_device, x_dict["params"]).cpu()
                del x_dict, z_dict

        else:
            conditions = conditions_batch.to(device)
            
            z_params_dict = model.encoder(torch.cat((x_device, conditions), dim=-1))

            for i in range(num_mc_epochs):
                if (k+1)*batch_size < data_size:
                    for param in z_rec["params"].keys(): z_rec["params"][param][k*batch_size:(k+1)*batch_size] = z_params_dict[param].cpu()
                    if (i+1)*mc_sample_batch_size < num_mc_samples:
                        z = model.encoder.sample(z_params_dict, num_samples=mc_sample_batch_size)
                        z_rec["samples"][i*mc_sample_batch_size:(i+1)*mc_sample_batch_size, k*batch_size:(k+1)*batch_size] = z
                        x_params_dict = model.decoder(torch.cat((z, conditions[None,...].repeat_interleave(mc_sample_batch_size, dim=0)), dim=-1))
                        for param in x_rec["params"].keys():
                            x_params_dict[param] = x_params_dict[param].view(mc_sample_batch_size, x_device.shape[0], -1)
                            x_rec["params"][param][i*mc_sample_batch_size:(i+1)*mc_sample_batch_size, k*batch_size:(k+1)*batch_size] = x_params_dict[param].cpu()
                        rlls[i*mc_sample_batch_size:(i+1)*mc_sample_batch_size, k*batch_size:(k+1)*batch_size] = model.decoder.log_likelihood(x_device, x_params_dict).cpu()
                    else:
                        z = model.encoder.sample(z_params_dict, num_samples=num_mc_samples-i*mc_sample_batch_size)
                        z_rec["samples"][i*mc_sample_batch_size:, k*batch_size:(k+1)*batch_size] = z
                        x_params_dict = model.decoder(torch.cat((z, conditions[None,...].repeat_interleave(num_mc_samples-i*mc_sample_batch_size, dim=0)), dim=-1))
                        for param in x_rec["params"].keys():
                            x_params_dict[param] = x_params_dict[param].view(num_mc_samples-i*mc_sample_batch_size, x_device.shape[0], -1)
                            x_rec["params"][param][i*mc_sample_batch_size:, k*batch_size:(k+1)*batch_size] = x_params_dict[param].cpu()
                        rlls[i*mc_sample_batch_size:, k*batch_size:(k+1)*batch_size] = model.decoder.log_likelihood(x_device, x_params_dict).cpu()
                else:
                    for param in z_rec["params"].keys(): z_rec["params"][param][k*batch_size:] = z_params_dict[param].cpu()
                    if (i+1)*mc_sample_batch_size < num_mc_samples:
                        z = model.encoder.sample(z_params_dict, num_samples=mc_sample_batch_size)
                        z_rec["samples"][i*mc_sample_batch_size:(i+1)*mc_sample_batch_size, k*batch_size:] = z
                        x_params_dict = model.decoder(torch.cat((z, conditions[None,...].repeat_interleave(mc_sample_batch_size, dim=0)), dim=-1))
                        for param in x_rec["params"].keys():
                            x_params_dict[param] = x_params_dict[param].view(mc_sample_batch_size, x_device.shape[0], -1)
                            x_rec["params"][param][i*mc_sample_batch_size:(i+1)*mc_sample_batch_size, k*batch_size:] = x_params_dict[param].cpu()
                        rlls[i*mc_sample_batch_size:(i+1)*mc_sample_batch_size, k*batch_size:] = model.decoder.log_likelihood(x_device, x_params_dict).cpu()
                    else:
                        z = model.encoder.sample(z_params_dict, num_samples=num_mc_samples-i*mc_sample_batch_size)
                        z_rec["samples"][i*mc_sample_batch_size:, k*batch_size:] = z
                        x_params_dict = model.decoder(torch.cat((z, conditions[None,...].repeat_interleave(num_mc_samples-i*mc_sample_batch_size, dim=0)), dim=-1))
                        for param in x_rec["params"].keys(): 
                            x_params_dict[param] = x_params_dict[param].view(num_mc_samples-i*mc_sample_batch_size, x_device.shape[0], -1)
                            x_rec["params"][param][i*mc_sample_batch_size:, k*batch_size:] = x_params_dict[param].cpu()
                        rlls[i*mc_sample_batch_size:, k*batch_size:] = model.decoder.log_likelihood(x_device, x_params_dict).cpu()
                del x_params_dict
            del z_params_dict

        if torch.cuda.is_available() and device!="cpu": torch.cuda.empty_cache()

    model.to("cpu")
    model.prior_params = {key: value.to("cpu") for key, value in model.prior_params.items()}
    return x_rec, z_rec, rlls






