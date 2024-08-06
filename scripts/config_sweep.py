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

def check_validity(combination, config):
    keys = []
    for key_path in combination.keys():
        keys.append(key_path.split('.'))

    # if "users" not in config["data"]["condition_tag_list"]:
    #     for key in keys:
    #         if "user_embedding_kwargs" in key: return False
    
    # if config["model"]["distribution_dict"]["likelihood"]["dist_type"] != "dict-gauss":
    #     for key in keys:
    #         if "vocab_size" in key: return False
    return True

def apply_rules(config, hyperparameters):
    hyperparameter_combinations = get_hyperparameters_combinations(hyperparameters)
    # mask = [True] * len(hyperparameter_combinations)
    # for i, combination in enumerate(hyperparameter_combinations):
    #     updated_config = copy.deepcopy(config)
    #     for key, value in combination.items(): update_config(updated_config, key, value)
    #     mask[i] = check_validity(combination, updated_config) # Check if the combination is valid
    # shifted_mask = [True] + mask[:-1]
    # memory_mask = [mask[i]|shifted_mask[i] for i in range(len(mask))]
    # hyperparameter_combinations = [hyperparameter_combinations[i] for i in range(len(mask)) if memory_mask[i]] # Too keep the first invalid combination for ignoring
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

    base_config["save_dir"] = "runs/sweep_runs"
    base_config["save_tag"] = "sweep_"

    hyperparameters = {
        "data": {
            "pad": [0],
            "ampute_params": {
                "a": [0.85],
                "b": [5, 10, 30]
            },
            "condition_tag_list": 
                [["months", "weekdays", "users"]],
            "user_embedding_kwargs": {
                "model_kwargs": {
                    "num_topics": [5, 20, 50],
                    "num_clusters": [100, 500, 1000]
                }
            }
        },
        "model": {
            "distribution_dict": {
                "likelihood": {
                    "dist_type": ["dict-gauss"],
                    "vocab_size": [100]
                }
            }
        }
    }

    run_sweep(base_config, hyperparameters)