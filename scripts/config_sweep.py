import json, pickle
import argparse
import os, sys
import itertools
import copy

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.training import load_config, main

def update_config(config, key_path, value):
    keys = key_path.split('.')
    d = config
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value

def generate_combinations(hyperparameters):
    keys, values = zip(*hyperparameters.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations

def expand_dict(d, prefix=''):
    items = []
    for k, v in d.items():
        new_key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            items.extend(expand_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_hyperparameters_combinations(hyperparameters):
    expanded_hyperparameters = expand_dict(hyperparameters)
    return generate_combinations(expanded_hyperparameters)

def check_validity(config):
    # Add checks here
    if config["model"]["distribution_dict"]["likelihood"]["num_neurons"] != config["model"]["distribution_dict"]["posterior"]["num_neurons"]: return False
    if config["model"]["distribution_dict"]["likelihood"]["num_hidden_layers"] != config["model"]["distribution_dict"]["posterior"]["num_hidden_layers"]: return False
    if config["model"]["distribution_dict"]["likelihood"]["dropout"] != config["model"]["distribution_dict"]["posterior"]["dropout"]: return False

    return True

def apply_rules(config, hyperparameters):
    hyperparameter_combinations = get_hyperparameters_combinations(hyperparameters)
    # for i, combination in enumerate(hyperparameter_combinations):
    #     updated_config = copy.deepcopy(config)
    #     for key, value in combination.items(): update_config(updated_config, key, value)

    return hyperparameter_combinations

def run_sweep(config, hyperparameters):
    hyperparameter_combinations = apply_rules(config, hyperparameters)
    for i, combination in enumerate(hyperparameter_combinations):
        updated_config = copy.deepcopy(config)
        for key, value in combination.items(): update_config(updated_config, key, value)
        if not check_validity(updated_config):
            print(f"INVALID COMBINATION!")
            print(f"Skipping combination {i + 1}/{len(hyperparameter_combinations)}: {combination}")
            continue
        print(f"Running combination {i + 1}/{len(hyperparameter_combinations)}: {combination}")
        main(updated_config)  # Assuming main() accepts a config dictionary

if __name__ == "__main__":
    config_path = './config_files/config0_gipuzkoa.json'
    base_config = load_config(config_path)

    base_config["save_dir"] = "runs/autoregressive_forecast/gipuzkoa"
    base_config["save_tag"] = "sweep_"

    hyperparameters = {
        "data": {
            # "random_seed": [100, 101],
            "pad": [0],
            "condition_tag_list":
                [["months", "weekdays", "day_befores", "users"]],
            "user_embedding_kwargs": {
                "model_kwargs": {
                    "num_topics": [50, 100, 500],
                    "num_clusters": [1000],
                    "scaling_per_user": [False],
                    # "reduce_dim": [True],
                    # "num_lower_dims": [5],
                }
            }
        },
        "model": {
            "latent_dim": [24],
            "distribution_dict": {
                "posterior": {
                },
                "likelihood": {
                    "dist_type": ["dict-gauss"],
                    "vocab_size": [100],
                }
            }
        },
        "train": {
            "lr_scheduling_kwargs": {
                # "min_lr": [5e-5]
            }
        }
    }

    run_sweep(base_config, hyperparameters)