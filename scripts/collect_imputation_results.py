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
import src.imputation_lib as imputation_lib


def main(args):
    folders = os.listdir(args.config_dir)
    config_file = "config.json"
    subfolder_name = "forecasting_results_"+str(args.unobserved_window_length)

    pbar = tqdm.tqdm(total=len(folders), dynamic_ncols=True)

    for i, folder in enumerate(folders):

        pbar.update(1)
        pbar.write(f"Imputing {folder} ({i+1}/{len(folders)})...")

        if not os.path.exists(os.path.join(args.config_dir, folder, "trained_model.pt")):
            pbar.write(f"Model file not found for {folder}. Skipping...")
            continue
        if not args.overwrite:
            if os.path.exists(os.path.join(args.config_dir, folder, subfolder_name, "test_results_aggregate.pkl")):
                pbar.write(f"Forecast results already exist for {folder}. Skipping...")
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

        if trainset.inputs.shape[-1]<args.unobserved_window_length:
            pbar.write(f"Unobserved window length is too long for {folder}. Skipping...")
            continue

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
                        # "test": X_test[:10],
                        "missing": utils.zero_preserved_log_normalize(X_missing*1.0, nonzero_mean, nonzero_std, log_output=log_space, zero_id=zero_id, shift=shift)}
        elif config["data"]["dataset_name"] == "STORM_daily":
            alpha = config["data"]["scaling"]["alpha"]

            pbar.write(f"Preparing datasets for {folder}...")

            inputs = {"train": trainset.inputs,
                        "val": valset.inputs,
                        "test": X_test,
                        "missing": X_missing}
            
        # condition_set["test"] = {key: value[:10] for key, value in condition_set["test"].items()}
        
        samples, results_raw, results_agg = {}, {}, {}

        if os.path.exists(os.path.join(args.config_dir, folder, subfolder_name, "test_samples.pkl")):
            pbar.write(f"Loading existing samples for {folder} as initial samples...")
            with open(os.path.join(args.config_dir, folder, subfolder_name, "test_samples.pkl"), 'rb') as f: missing_data_init = pickle.load(f)
        else: missing_data_init = {k: None for k in inputs.keys()}
        
        sets_to_investigate = ["test"]

        for set_type in sets_to_investigate:
            results_raw[set_type], results_agg[set_type] =  {}, {}

            x_missing = inputs[set_type].copy()
            x_missing[:,-args.unobserved_window_length:] = np.nan

            pbar.write(f"Forecasting {set_type}set by imputing...")

            loglikelihoods, samples[set_type] = imputation_lib.mass_cvae_imputation_with_loglikelihood(model, x_missing, inputs[set_type][:,-args.unobserved_window_length:], conditioner, condition_set[set_type], missing_data_init=missing_data_init[set_type], batch_size=args.batch_size, num_samples=args.num_imputation_samples, warmup_steps=args.num_pseudo_gibbs_steps, num_iter=args.num_metropolis_within_hasting_steps, verbose_freq=100, device=args.device)

            print(f"Calculating errors...")
            p = [1, 2]

            errors = {}

            for p_val in p:
                print(f"Calculating l_{p_val} error...")
                if p_val==np.inf: key = "l_inf"
                else: key = f"l_{p_val}"
                errors[key] = np.mean((np.abs(samples[set_type] - inputs[set_type][:,-args.unobserved_window_length:][None,...])**p_val).sum(axis=-1), axis=0)**(1/p_val)

            df = pd.DataFrame(errors)
            df["loglikelihood"] = loglikelihoods
            df["user_id"] = user_ids[set_type]
            df["month"] = months[set_type]
            # df["user_id"] = user_ids[set_type][:10]
            # df["month"] = months[set_type][:10]

            # df = pd.DataFrame({"loglikelihood": loglikelihoods, "user_id": user_ids[set_type], "month": months[set_type]})
            # df = pd.DataFrame({"loglikelihood": loglikelihoods, "user_id": user_ids[set_type][:10], "month": months[set_type][:10]})

            # results_agg[set_type]["users"] = df.groupby("user_id").mean().values[:,0]
            # results_agg[set_type]["months"] = df.groupby("month").mean().values[:,0]
            # results_agg[set_type]["loglikelihood"] = loglikelihoods.mean().item()

            # results_raw[set_type] = df
            # results_raw[set_type]["set_type"] = set_type

            results_agg[set_type]["loglikelihood"] = loglikelihoods.mean().item()
            for key in errors.keys():
                results_agg[set_type]["error_"+key] = errors[key].mean().item()

            results_raw[set_type] = df
            results_raw[set_type]["set_type"] = set_type


        pbar.write("Saving results...")
        if not os.path.exists(os.path.join(args.config_dir, folder, subfolder_name)): os.makedirs(os.path.join(args.config_dir, folder, subfolder_name))
            
        results_concat = pd.concat([results_raw[set_type] for set_type in sets_to_investigate])
        results_concat.to_csv(os.path.join(args.config_dir, folder, subfolder_name, "test_results_raw.csv"), index=False)

        with open(os.path.join(args.config_dir, folder, subfolder_name, "test_results_aggregate.pkl"), 'wb') as f: pickle.dump(results_agg, f)

        with open(os.path.join(args.config_dir, folder, subfolder_name, "test_results_samples.pkl"), 'wb') as f: 
            pickle.dump({k:v[args.num_imputation_samples%(args.num_pseudo_gibbs_steps+args.num_metropolis_within_hasting_steps)-1] for k,v in samples.items()}, f)

        pbar.write(f"Finished testing {folder} ({i+1}/{len(folders)})")

    pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test all the models in a directory.")
    parser.add_argument("--config_dir", type=str, default="runs/imputation_forecast/gipuzkoa", help="Path to the directory containing the saved model folders.")
    parser.add_argument("--overwrite", type=utils.str2bool, default=True, help="Whether to overwrite existing test results.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the models on.")
    parser.add_argument("--num_imputation_samples", type=int, default=100, help="Number of samples for imputation.")
    parser.add_argument("--batch_size", type=int, default=25000, help="Batch size for reconstruction and imputation.")
    parser.add_argument("--unobserved_window_length", type=int, default=24, help="Number of unobserved time steps (from the end) to impute.")
    parser.add_argument("--num_pseudo_gibbs_steps", type=int, default=1000, help="Number of pseudo gibbs (warm-up) steps.")
    parser.add_argument("--num_metropolis_within_hasting_steps", type=int, default=0, help="Number of metropolis within hasting steps.")

    args = parser.parse_args()
    main(args)