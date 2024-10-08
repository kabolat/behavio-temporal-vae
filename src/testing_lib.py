import torch
import numpy as np

@torch.no_grad()
def mass_loglikelihood(model, x, conditions, num_mc_samples=1, batch_size=10000, mc_sample_batch_size=100, device="cpu"):

    _, z_rec, rlls = mass_reconstruction(model, x, conditions, num_mc_samples=num_mc_samples, batch_size=batch_size, mc_sample_batch_size=mc_sample_batch_size, device=device)

    posterior_loglikelihood = model.encoder.log_likelihood(z_rec["samples"], z_rec["params"])
    prior_loglikelihood = model.encoder.log_likelihood(z_rec["samples"], model.prior_params)
    kl_divergence = model.kl_divergence(z_rec["params"], prior_params=model.prior_params)

    loglikelihood = -torch.log(torch.tensor(rlls.shape[0])) + torch.logsumexp(-posterior_loglikelihood + rlls + prior_loglikelihood, dim=0)

    return loglikelihood.numpy(), rlls.numpy(), kl_divergence.numpy()


@torch.no_grad()
def mass_reconstruction(model, x, conditions, num_mc_samples=1, batch_size=10000, mc_sample_batch_size=100, device="cpu"):
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

        print(f"Batch {k+1}/{len(dataloader)}")

    model.to("cpu")
    model.prior_params = {key: value.to("cpu") for key, value in model.prior_params.items()}
    return x_rec, z_rec, rlls






