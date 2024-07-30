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

def check_validity(config, key_path):
    keys = key_path.split('.')
    # Define your rules here
    if "users" not in config["data"]["condition_tag_list"]:
        if keys[1] == "user_embedding_kwargs": return False
    if config["model"]["distribution_dict"]["likelihood"]["dist_type"] == "dict-gauss":
        if keys[1] == "vocab_size": return False
        if config["data"]["pad"]*2+24 >= config["model"]["distribution_dict"]["likelihood"]["vocab_size"]: 
            return False
    return True

def apply_rules(config, key_path):
    hyperparameter_combinations = get_hyperparameters_combinations(hyperparameters)
    for i, combination in enumerate(hyperparameter_combinations):
        updated_config = copy.deepcopy(config)
        for key, value in combination.items():
            update_config(updated_config, key, value)
            if not check_validity(updated_config, key): hyperparameter_combinations.pop(i)
    return hyperparameter_combinations

def run_sweep(config, hyperparameters):
    hyperparameter_combinations = apply_rules(config, hyperparameters)
    for i, combination in enumerate(hyperparameter_combinations):
        updated_config = copy.deepcopy(config)
        for key, value in combination.items(): update_config(updated_config, key, value)
        print(f"Running combination {i + 1}/{len(hyperparameter_combinations)}: {combination}")
        main(updated_config)  # Assuming main() accepts a config dictionary

if __name__ == "__main__":
    config_path = './config_files/config0.json'
    base_config = load_config(config_path)

    base_config["save_dir"] = "sweep_runs"
    base_config["save_tag"] = "sweep_"

    hyperparameters = {
        "data": {
            "pad": [0],
            "ampute_params": {
                "a": [0.85],
                "b": [5, 10, 30]
            },
            "condition_tag_list": 
                [["months", "weekdays", "users"], ["months", "weekdays"]],
            "user_embedding_kwargs": {
                "model_kwargs": {
                    "num_topics": [5, 10, 20, 40],
                    "num_clusters": [500, 1000, 2000]
                }
            }
        },
        "model": {
            "distribution_dict": {
                "likelihood": {
                    "dist_type": ["normal", "dict-gauss"],
                    "vocab_size": [50, 100, 200]
                }
            }
        }
    }

    run_sweep(base_config, hyperparameters)