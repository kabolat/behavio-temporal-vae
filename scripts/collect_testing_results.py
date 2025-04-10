import os, sys
import argparse
import json
import pandas as pd
import numpy as np
import torch
import tqdm
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.vae_models import CVAE
import src.utils as utils
import src.preprocess_lib as preprocess_lib
import src.testing_lib as testing_lib

def main(args):
    folders = os.listdir(args.config_dir)
    config_file = "config.json"

    pbar = tqdm.tqdm(total=len(folders), dynamic_ncols=True)

    for i, folder in enumerate(folders):

        pbar.update(1)
        pbar.write(f"Testing {folder} ({i+1}/{len(folders)})...")

        if not os.path.exists(os.path.join(args.config_dir, folder, "trained_model.pt")):
            pbar.write(f"Model file not found for {folder}. Skipping...")
            continue
        if not args.overwrite:
            if os.path.exists(os.path.join(args.config_dir, folder, "test_results_raw.csv")) and os.path.exists(os.path.join(args.config_dir, folder, "test_results_aggregate.pkl")):
                pbar.write(f"Test results already exist for {folder}. Skipping...")
                continue
            
        # Load config file
        pbar.write(f"Loading config file for {folder}...")
        if not os.path.exists(os.path.join(args.config_dir, folder, config_file)):
            pbar.write(f"Config file not found for {folder}. Skipping...")
            continue
        with open(os.path.join(args.config_dir, folder, config_file), 'r') as f: config = json.load(f)
        
        ## Load the data
        # utils.blockPrint()
        trainset, valset, conditioner, user_ids, months, years, indices, condition_set, X_test, X_missing, _, nonzero_mean, nonzero_std = preprocess_lib.prepare_data(config["data"])

        # Load model
        model = CVAE(input_dim=trainset.inputs.shape[1], conditioner=conditioner, **config["model"])
        model.load(os.path.join(args.config_dir, folder))
        model.eval()
        # utils.enablePrint()
        
        #Prepare the datasets
        if config["data"]["dataset_name"] == "goi4_dp_full_Gipuzkoa":
            log_space = config["data"]["scaling"]["log_space"]
            zero_id = config["data"]["scaling"]["zero_id"]
            shift = config["data"]["scaling"]["shift"]
            
            pbar.write(f"Preparing datasets for {folder}...")

            inputs = {"train": trainset.inputs,
                        "val": valset.inputs,
                        "test": X_test,
                        "missing": utils.zero_preserved_log_normalize(X_missing*1.0, nonzero_mean, nonzero_std, log_output=log_space, zero_id=zero_id, shift=shift)}
        elif config["data"]["dataset_name"] == "STORM_daily":
            alpha = config["data"]["scaling"]["alpha"]

            pbar.write(f"Preparing datasets for {folder}...")

            inputs = {"train": trainset.inputs,
                        "val": valset.inputs,
                        "test": X_test,
                        "missing": X_missing}
        
        results_raw, results_agg = {}, {}
        
        sets_to_investigate = ["test"]

        for set_type in sets_to_investigate:
            results_raw[set_type], results_agg[set_type] = {}, {}

            print(f"Testing {set_type} set...")
            print(f"Mass reconstructing...")
            x_rec, z_rec, rlls = testing_lib.mass_reconstruction(model, inputs[set_type], conditioner, condition_set[set_type], num_mc_samples=args.num_rec_samples, batch_size=args.batch_size, mc_sample_batch_size=args.batch_size_mc, device=args.device)

            print(f"Calculating loglikelihood...")
            with torch.no_grad():
                posterior_loglikelihood = model.encoder.log_likelihood(z_rec["samples"], z_rec["params"])
                prior_loglikelihood = model.encoder.log_likelihood(z_rec["samples"], model.prior_params)
                kl_divergence = model.kl_divergence(z_rec["params"], prior_params=model.prior_params)

            loglikelihood = -torch.log(torch.tensor(rlls.shape[0])) + torch.logsumexp(-posterior_loglikelihood + rlls + prior_loglikelihood, dim=0)
            loglikelihood = loglikelihood.numpy()
            rlls = rlls.numpy()
            kl_divergence = kl_divergence.numpy()

            del z_rec, rlls

            print(f"Sampling...")
            x_samples = model.decoder.sample(x_rec["params"])[0].cpu().detach().numpy()

            del x_rec

            print(f"Calculating errors...")
            p = [1, 2]

            errors = {}

            for p_val in p:
                print(f"Calculating l_{p_val} error...")
                if p_val==np.inf: key = "l_inf"
                else: key = f"l_{p_val}"
                errors[key] = np.mean((np.abs(x_samples - inputs[set_type][None,...])**p_val).sum(axis=-1), axis=0)**(1/p_val)

            # loglikelihood, _, _ = testing_lib.mass_loglikelihood(model, inputs[set_type], conditioner, condition_set[set_type], num_mc_samples=args.num_rec_samples, batch_size=args.batch_size, mc_sample_batch_size=args.batch_size_mc, device=args.device)

            df = pd.DataFrame(errors)
            df["loglikelihood"] = loglikelihood
            df["user_id"] = user_ids[set_type]
            df["month"] = months[set_type]
            # df = pd.DataFrame({"loglikelihood": loglikelihood, "user_id": user_ids[set_type], "month": months[set_type]})

            # results_agg[set_type]["users"] = df.groupby("user_id").mean().values[:,0]
            # results_agg[set_type]["months"] = df.groupby("month").mean().values[:,0]
            results_agg[set_type]["loglikelihood"] = loglikelihood.mean().item()
            for key in errors.keys():
                results_agg[set_type]["error_"+key] = errors[key].mean().item()

            results_raw[set_type] = df
            results_raw[set_type]["set_type"] = set_type


        pbar.write("Saving results...")
        results_concat = pd.concat([results_raw[set_type] for set_type in sets_to_investigate])
        results_concat.to_csv(os.path.join(args.config_dir, folder, "test_results_raw.csv"), index=False)

        with open(os.path.join(args.config_dir, folder, "test_results_aggregate.pkl"), 'wb') as f: pickle.dump(results_agg, f)

        pbar.write(f"Finished testing {folder} ({i+1}/{len(folders)})")

    pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test all the models in a directory.")
    parser.add_argument("--config_dir", type=str, default="runs/autoregressive_forecast/gipuzkoa", help="Path to the directory containing the saved model folders.")
    parser.add_argument("--overwrite", type=utils.str2bool, default=True, help="Whether to overwrite existing test results.")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to run the models on.")
    parser.add_argument("--num_rec_samples", type=int, default=100, help="Number of posterior samples for reconstruction.")
    parser.add_argument("--batch_size", type=int, default=50000, help="Batch size for reconstruction and imputation.")
    parser.add_argument("--batch_size_mc", type=int, default=40, help="Batch size for prior Monte Carlo samples (for scalability).")

    args = parser.parse_args()
    main(args)