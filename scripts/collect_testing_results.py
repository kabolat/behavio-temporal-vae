import os, sys
import argparse
import json
import pandas as pd
import numpy as np
import torch
import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.vae_models import CVAE
import src.utils as utils
import src.preprocess_lib as preprocess_lib
import src.testing_lib as testing_lib

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def main(args):
    folders = os.listdir(args.config_dir)
    config_file = "config.json"

    pbar = tqdm.tqdm(total=len(folders), dynamic_ncols=True)

    for i, folder in enumerate(folders):

        pbar.update(1)
        pbar.write(f"Testing {folder} ({i+1}/{len(folders)})...")

        if not args.overwrite:
            if os.path.exists(os.path.join(args.config_dir, folder, "test_results.json")):
                pbar.write(f"Test results already exist for {folder}. Skipping...")
                continue

        test_results = {}
        # Load config file
        pbar.write(f"Loading config file for {folder}...")
        if not os.path.exists(os.path.join(args.config_dir, folder, config_file)):
            pbar.write(f"Config file not found for {folder}. Skipping...")
            continue
        with open(os.path.join(args.config_dir, folder, config_file), 'r') as f: config = json.load(f)
        
        blockPrint()
        _, valset, conditioner, _, condition_set, X_test, num_missing_days, nonzero_mean, nonzero_std = preprocess_lib.prepare_data(config["data"])
        num_users = len(num_missing_days)

        # Load model
        model = CVAE(input_dim=valset.inputs.shape[1], conditioner=conditioner, **config["model"])
        enablePrint()

        if not os.path.exists(os.path.join(args.config_dir, folder, "model.pth")):
            pbar.write(f"Model file not found for {folder}. Skipping...")
            continue
        model.load(os.path.join(args.config_dir, folder))
        
        log_space = config["data"]["scaling"]["log_space"]
        zero_id = config["data"]["scaling"]["zero_id"]
        shift = config["data"]["scaling"]["shift"]
        
        pbar.write(f"Preparing test data for {folder}...")
        x_test = utils.zero_preserved_log_normalize(X_test*1.0, nonzero_mean, nonzero_std, log_output=log_space, zero_id=zero_id, shift=shift)
        x_test = torch.tensor(x_test).float()
        conditions_test =  torch.tensor(conditioner.transform(condition_set["test"].copy())).float()
        
        pbar.write("Reconstructing...")
        x_rec, z_rec = testing_lib.mass_reconstruction(model, x_test, conditions_test, num_mc_samples=args.num_rec_samples, batch_size=args.batch_size, device=args.device)

        pbar.write("Calculating probabilistic metrics...")
        test_results["prob_metrics"] = testing_lib.get_probabilistic_metrics(model, x_test, x_rec, z_rec, aggregate=True, device=args.device)

        pbar.write("Imputing...")
        x_imp = testing_lib.mass_imputation(model, conditions_test, num_mc_samples_prior=args.num_imp_samples_prior, num_mc_samples_likelihood=args.num_imp_samples_likelihood, batch_size=args.batch_size, device=args.device)

        x_imp_denormalized = testing_lib.mass_denormalization(model=model, x_imp=x_imp, nonzero_mean=nonzero_mean, nonzero_std=nonzero_std, zero_id=zero_id, shift=shift, log_space=log_space, device=args.device)

        pbar.write("Calculating sample metrics...")
        test_results["sample_metrics"] = {}
        for imputation_style in ["samples", "mean"]:
            test_results["sample_metrics"][imputation_style] = {}
            test_results["sample_metrics"][imputation_style] = testing_lib.get_sample_metrics(x_test, x_imp_denormalized, imputation_style=imputation_style, aggregate=True)

        pbar.write("Saving results...")
        with open(os.path.join(args.config_dir, folder, "test_results.json"), 'w') as f:
            json.dump(test_results, f, indent=4)

        pbar.write(f"Finished testing {folder} ({i+1}/{len(folders)})")

    pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test all the models in a directory.")
    parser.add_argument("--config_dir", type=str, default="runs/sweep_runs_corrected", help="Path to the directory containing the saved model folders.")
    parser.add_argument("--overwrite", type=bool, default=False, help="Whether to overwrite existing test results.")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to run the models on.")
    parser.add_argument("--num_rec_samples", type=int, default=100, help="Number of posterior samples for reconstruction.")
    parser.add_argument("--num_imp_samples_prior", type=int, default=20, help="Number of (reconstruction-free) imputation samples (prior).")
    parser.add_argument("--num_imp_samples_likelihood", type=int, default=20, help="Number of (reconstruction-free) imputation samples (likelihood).")
    parser.add_argument("--batch_size", type=int, default=20000, help="Batch size for reconstruction and imputation.")

    args = parser.parse_args()
    main(args)