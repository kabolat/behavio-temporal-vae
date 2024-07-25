import json, pickle
import argparse
import os, sys

import numpy as np
import pandas as pd
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.vae_models import VAE, CVAE
import src.datasets as datasets
import src.utils as utils
import src.preprocess_lib as preprocess_lib
import src.conditioning_lib as conditioning_lib
from src.user_encoding_lib import UserEncoder


def load_config(json_file):
    with open(json_file, 'r') as file: return json.load(file)

def prepare_data(config_data):
    RANDOM_SEED = config_data["random_seed"]
    DATASET_DIR = config_data["dataset_dir"]
    DATASET_NAME = config_data["dataset_name"]
    CONDITION_TAG_LIST = config_data["condition_tag_list"]
    RESOLUTION = config_data["resolution"] #in hours
    PAD = config_data["pad"] #in features
    AMPUTE_PARAMS = config_data["ampute_params"]
    USER_SUBSAMPLE_RATE, DAY_SUBSAMPLE_RATE = config_data["subsample_rate"]["user"], config_data["subsample_rate"]["day"]
    SHIFT = config_data["scaling"]["shift"]
    ZERO_ID = config_data["scaling"]["zero_id"]
    LOG_SPACE = config_data["scaling"]["log_space"]
    VAL_RATIO = config_data["val_ratio"]
    USER_EMBEDDING_KWARGS = config_data["user_embedding_kwargs"]
    NUM_TOPICS = USER_EMBEDDING_KWARGS["model_kwargs"]["num_topics"]
    VOCAB_SIZE = USER_EMBEDDING_KWARGS["model_kwargs"]["num_clusters"]
    USER_EMBEDDING_KWARGS["model_kwargs"]["random_state"] = RANDOM_SEED
    USER_EMBEDDING_KWARGS["model_kwargs"]["user_subsample_rate"]: USER_SUBSAMPLE_RATE
    USER_EMBEDDING_KWARGS["fit_kwargs"]["lda"]["doc_topic_prior"] = 1.0/NUM_TOPICS
    USER_EMBEDDING_KWARGS["fit_kwargs"]["lda"]["topic_word_prior"] = 1.0/VOCAB_SIZE

    ## Import data
    dataset_path = os.path.join(DATASET_DIR, DATASET_NAME)
    df = pd.read_csv(os.path.join(dataset_path, 'dataset.csv'))
    data, dates = df.iloc[:,:-2].values, df.date.values
    num_days, num_users = df.date.nunique(), df.user.nunique()
    print(f'Dataset: {DATASET_NAME}')
    print(f'Loaded {len(data)} consumption profiles from {num_days} dates and {num_users} users.')

    date_dict = np.load(os.path.join(dataset_path, 'encode_dict.npy'), allow_pickle=True).item()["date_dict"]
    date_dict_inv = {v: k for k, v in date_dict.items()}
    if not os.path.exists(os.path.join(dataset_path, 'raw_dates.npy')):
        raw_dates = np.array([datetime.datetime.strptime(date_dict_inv[d], '%Y-%m-%d') for d in dates])
        np.save(os.path.join(dataset_path, 'raw_dates.npy'), raw_dates)
    else: raw_dates = np.load(os.path.join(dataset_path, 'raw_dates.npy'), allow_pickle=True)
    metadata = pd.read_csv(os.path.join(dataset_path, 'metadata.csv'))
    unique_provinces = metadata.province.unique()
    print(f'Loaded metadata for {len(unique_provinces)} provinces')
    print(f"Uniqe provinces are: {unique_provinces}")

    ## Preprocess data

    ## Set resolution and pad
    X = preprocess_lib.downsample_and_pad(np.reshape(data, (num_users, num_days, -1)), RESOLUTION, PAD)
    X, user_mask = preprocess_lib.remove_unwanted_profiles(X)
    X, raw_dates = preprocess_lib.subsample_data(X, np.reshape(raw_dates, (num_users, num_days))[user_mask], USER_SUBSAMPLE_RATE, DAY_SUBSAMPLE_RATE)

    num_users, num_days, num_features = X.shape

    print("{:.<40}{:.>5}".format("Number of (subsampled/filtered) users", num_users))
    print("{:.<40}{:.>5}".format("Number of (subsampled) days", num_days))
    print("{:.<40}{:.>5}".format("Number of (aggregated) features", num_features))

    X_missing, missing_idx, num_missing_profiles, missing_days = preprocess_lib.ampute_data(X, a=AMPUTE_PARAMS["a"], b=AMPUTE_PARAMS["b"], random_seed=RANDOM_SEED)
    
    nonzero_mean, nonzero_std = utils.zero_preserved_log_stats(X_missing)
    X_missing = utils.zero_preserved_log_normalize(X_missing, nonzero_mean, nonzero_std, log_output=LOG_SPACE, zero_id=ZERO_ID, shift=SHIFT)

    condition_kwargs, condition_set = conditioning_lib.prepare_conditions(CONDITION_TAG_LIST, raw_dates, data=X_missing.reshape(num_users, num_days, -1), dataset_path=dataset_path, user_embedding_kwargs=USER_EMBEDDING_KWARGS, config_dict=config_data)

    X_observed, user_ids_observed, condition_set_observed, X_test_flat, X_test_list, user_ids_test, condition_set_test = preprocess_lib.separate_test_set(X, X_missing, condition_set, missing_idx, missing_days)

    X_train, user_ids_train, conditions_train, X_val, user_ids_val, conditions_val = preprocess_lib.separate_val_set(X_observed, user_ids_observed, condition_set_observed, val_ratio=VAL_RATIO, random_seed=RANDOM_SEED)

    conditioner = conditioning_lib.Conditioner(**condition_kwargs, condition_set=condition_set_observed)
    trainset = datasets.ConditionedDataset(inputs=X_train, conditions=conditions_train, conditioner=conditioner)
    valset = datasets.ConditionedDataset(inputs=X_val, conditions=conditions_val, conditioner=conditioner)
    print(f"Number of Training Points: {len(trainset)}")
    print(f"Number of Validation Points: {len(valset)}")

    return trainset, valset, conditioner

def train_model(model, trainset, valset, train_kwargs, writer=None):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_kwargs["batch_size"], shuffle=True, drop_last=True, num_workers=1, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=8196, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    model.fit(trainloader=trainloader, valloader=valloader, **train_kwargs, writer=writer)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    model.to("cpu")
    model.prior_params = {k: v.to("cpu") for k, v in model.prior_params.items()}
    model.eval()
    return model


def main(args):
    config = load_config(args.config_path)

    trainset, valset, conditioner = prepare_data(config["data"])

    model = CVAE(input_dim=trainset.inputs.shape[-1], conditioner=conditioner, **config["model"])
    print("Number of encoder parameters:", model.encoder._num_parameters())
    print("Number of decoder parameters:", model.decoder._num_parameters())

    if config["save_dir"] is not None:
        current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        writer = SummaryWriter(log_dir=os.path.join(config["save_dir"], config["save_tag"]+current_time))
    else: writer = SummaryWriter()

    with open(os.path.join(writer.log_dir, 'config.json'), 'w') as file: json.dump(config, file, indent=4)

    model = train_model(model, trainset, valset, config["train"], writer=writer)
    model.save()
    conditioner.save(model.log_dir)
    print(f"Model saved at {model.log_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a (user-informed) customisable VAE with the configuration from a JSON file.")
    parser.add_argument("--config_path", type=str, default="./config_files/config0.json", help="Path to the JSON configuration file.")
    args = parser.parse_args()
    main(args)