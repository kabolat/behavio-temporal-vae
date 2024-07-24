import json, pickle
import argparse
import os, sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.vae_models import VAE, CVAE
import src.datasets as datasets
import src.utils as utils
from src.lda_lib import EntityEncoder

def load_config(json_file):
    with open(json_file, 'r') as file:
        return json.load(file)

def prepare_data(config):
    RANDOM_SEED = config["random_seed"]
    DATASET_DIR = config["data"]["dataset_dir"]
    DATASET_NAME = config["data"]["dataset_name"]
    ADD_MONTHS = config["data"]["conditions"]["add_months"]
    ADD_WEEKDAYS = config["data"]["conditions"]["add_weekdays"]
    ADD_IS_WEEKEND = config["data"]["conditions"]["add_is_weekend"]
    ADD_TEMPERATURE_MIN = config["data"]["conditions"]["add_temperature_min"]
    ADD_TEMPERATURE_MAXDELTA = config["data"]["conditions"]["add_temperature_maxdelta"]
    ADD_PRECIPITATION_LEVEL = config["data"]["conditions"]["add_precipitation_level"]
    ADD_USERS = config["data"]["conditions"]["add_users"]
    RESOLUTION = config["data"]["resolution"] #in hours
    PAD = config["data"]["pad"] #in hours
    AMPUTE_PARAMS = config["data"]["ampute_params"]
    USER_SUBSAMPLE_RATE, DAY_SUBSAMPLE_RATE = config["data"]["subsample_rate"]["user"], config["data"]["subsample_rate"]["day"]
    SHIFT = config["data"]["scaling"]["shift"]
    ZERO_ID = config["data"]["scaling"]["zero_id"]
    LOG_SPACE = config["data"]["scaling"]["log_space"]
    VAL_RATIO = config["data"]["val_ratio"]
    USER_EMBEDDING_KWARGS = config["data"]["user_embedding_kwargs"]
    NUM_TOPICS = USER_EMBEDDING_KWARGS["model_kwargs"]["num_topics"]
    VOCAB_SIZE = USER_EMBEDDING_KWARGS["model_kwargs"]["num_clusters"]
    USER_EMBEDDING_KWARGS["model_kwargs"]["random_state"] = RANDOM_SEED
    USER_EMBEDDING_KWARGS["model_kwargs"]["user_subsample_rate"]: USER_SUBSAMPLE_RATE
    USER_EMBEDDING_KWARGS["fit_kwargs"]["doc_topic_prior"] = 1.0/NUM_TOPICS
    USER_EMBEDDING_KWARGS["fit_kwargs"]["topic_word_prior"] = 1.0/VOCAB_SIZE

    ## Import data
    dataset_path = f'{DATASET_DIR}/{DATASET_NAME}'
    df = pd.read_csv(f'{dataset_path}/dataset.csv')
    data, dates, users = df.iloc[:,:-2].values, df.date.values, df.user.values
    date_ids, user_ids = df.date.unique(), df.user.unique()
    num_days, num_users = len(date_ids), len(user_ids)
    print(f'Loaded {len(data)} consumption profiles from {num_days} dates and {num_users} users')

    date_dict = np.load(f'{dataset_path}/encode_dict.npy', allow_pickle=True).item()["date_dict"]
    date_dict_inv = {v: k for k, v in date_dict.items()}

    if not os.path.exists(f'{dataset_path}/raw_dates.npy'):
        import datetime
        raw_dates = np.array([datetime.datetime.strptime(date_dict_inv[d], '%Y-%m-%d') for d in dates])
        np.save(f'{dataset_path}/raw_dates.npy', raw_dates)
    else:raw_dates = np.load(f'{dataset_path}/raw_dates.npy', allow_pickle=True)
    metadata = pd.read_csv(f'{dataset_path}/metadata.csv')
    unique_provinces = metadata.province.unique()
    print(f'Loaded metadata for {len(unique_provinces)} provinces')
    print(f"Uniqe provinces are: {unique_provinces}")

    ## Prepare conditions
    months = np.array([d.month for d in raw_dates])
    weekdays = np.array([d.weekday() for d in raw_dates])
    is_weekend = np.array([int(d.weekday() >= 5) for d in raw_dates])

    df_temp = pd.read_csv(f'{dataset_path}/spain_temp_daily.csv')
    df_temp.index = pd.to_datetime(df_temp['date'])
    df_temp.drop(columns='date', inplace=True)
    df_temp = df_temp.loc[raw_dates]

    df_prec = pd.read_csv(f'{dataset_path}/spain_prec_daily.csv')
    df_prec.index = pd.to_datetime(df_prec['date'])
    df_prec.drop(columns='date', inplace=True)
    df_prec = df_prec.loc[raw_dates]
    df_prec = df_prec.sort_values(by='prec_total')

    condition_kwargs = {}

    condition_kwargs["tags"], condition_kwargs["types"], condition_kwargs["supports"], condition_set  = [], [], [], {}
    if ADD_MONTHS: 
        condition_kwargs["tags"].append("months")
        condition_kwargs["types"].append("circ")
        condition_kwargs["supports"].append(np.unique(months).tolist())
        condition_set["months"] = months[...,None]
    if ADD_WEEKDAYS: 
        condition_kwargs["tags"].append("weekdays")
        condition_kwargs["types"].append("circ")
        condition_kwargs["supports"].append(np.unique(weekdays).tolist())
        condition_set["weekdays"] = weekdays[...,None]
    if ADD_IS_WEEKEND:
        condition_kwargs["tags"].append("is_weekend")
        condition_kwargs["types"].append("cat")
        condition_kwargs["supports"].append([0, 1])
        condition_set["is_weekend"] = is_weekend[...,None]
    if ADD_TEMPERATURE_MIN:
        condition_kwargs["tags"].append("temp_min")
        condition_kwargs["types"].append("cont")
        condition_kwargs["supports"].append([df_temp[condition_kwargs["tags"][-1]].min(), df_temp[condition_kwargs["tags"][-1]].max()])
        condition_set["temp_min"] = df_temp[condition_kwargs["tags"][-1]].values[...,None]
    if ADD_TEMPERATURE_MAXDELTA:
        condition_kwargs["tags"].append("temp_max_delta")
        condition_kwargs["types"].append("cont")
        condition_kwargs["supports"].append([df_temp[condition_kwargs["tags"][-1]].min(), df_temp[condition_kwargs["tags"][-1]].max()])
        condition_set["temp_max_delta"] = df_temp[condition_kwargs["tags"][-1]].values[...,None]
    if ADD_PRECIPITATION_LEVEL:
        condition_kwargs["tags"].append("precipitation_level")
        condition_kwargs["types"].append("ord")
        condition_kwargs["supports"].append(np.unique(df_prec["label"]).tolist())
        condition_set["precipitation_level"] = df_prec["label"].values[...,None]
    
    conditioner = datasets.Conditioner(**condition_kwargs, condition_set=condition_set)

    ## Set resolution and pad
    if RESOLUTION == 12:
        X = np.reshape(data, (-1, 24))
        X = np.reshape(np.concatenate([X[:,6:], X[:,:6]], axis=-1), (num_users, num_days, int(24/RESOLUTION), int(RESOLUTION))).sum(axis=-1)    #circle shift the last dimension of X
    else:
        X = np.reshape(data, (num_users, num_days, int(24/RESOLUTION), int(RESOLUTION))).sum(axis=-1)
        if PAD != 0: X = np.concatenate((X[:,:-(PAD//24+2),-PAD:], X[:,(PAD//24+1):-(PAD//24+1),:], X[:,(PAD//24+2):,:PAD]), axis=-1)

    condition_set = {k: np.reshape(v, (num_users, num_days, -1)) for k, v in condition_set.items()}
    if PAD != 0: condition_set = {k: v[:,1:-1,:] for k, v in condition_set.items()}
    num_days = X.shape[1]

    ## Clean data
    nonzero_user_mask = np.sum(np.all(X == 0, axis=2), axis=1) < num_days
    print(f'Removing {(~nonzero_user_mask).sum()} users with all-zero consumption profiles')
    positive_user_mask = np.sum(np.any(X < 0, axis=2), axis=1) == 0
    print(f'Removing {(~positive_user_mask).sum()} users with any-negative consumption profiles')
    user_mask = nonzero_user_mask & positive_user_mask
    X = X[user_mask]
    condition_set = {k: v[user_mask] for k, v in condition_set.items()}

    ## Ampute the dataset
    np.random.seed(RANDOM_SEED)
    n, a, b = num_days, AMPUTE_PARAMS["a"], AMPUTE_PARAMS["b"]
    missing_days = np.random.binomial(n, p=np.random.beta(a, b, size=X.shape[0]), size=X.shape[0])
    print(f"Mean of missing days: {n*a/(a+b):.2f}")

    X_missing = X.copy().astype(float)
    condition_missing = {k: v.copy().astype(float) for k, v in condition_set.items()}

    for user in range(X.shape[0]): 
        X_missing[user, :missing_days[user]] = np.nan
        for k in condition_missing.keys():
            condition_missing[k][user, :missing_days[user]] = np.nan
    
    ## Subsample the dataset
    X, X_missing = X[::USER_SUBSAMPLE_RATE, ::DAY_SUBSAMPLE_RATE, :], X_missing[::USER_SUBSAMPLE_RATE, ::DAY_SUBSAMPLE_RATE, :]
    condition_set = {k: v[::USER_SUBSAMPLE_RATE, ::DAY_SUBSAMPLE_RATE, :] for k, v in condition_set.items()}
    condition_missing = {k: v[::USER_SUBSAMPLE_RATE, ::DAY_SUBSAMPLE_RATE, :] for k, v in condition_missing.items()}
    num_users, num_days, num_features = X.shape
    X_gt_list = [X[user, :missing_days[user]]*1 for user in range(num_users)]
    X_gt_condition_list = {k: [v[user, :missing_days[user]]*1 for user in range(num_users)] for k, v in condition_set.items()}

    print("{:.<40}{:.>5}".format("Number of (subsampled/filtered) users", num_users))
    print("{:.<40}{:.>5}".format("Number of (subsampled) days", num_days))
    print("{:.<40}{:.>5}".format("Number of (aggregated) features", num_features))

    missing_idx_mat  = np.isnan(X_missing).any(2)
    missing_num_labels = {"user": missing_idx_mat.sum(1), "day": missing_idx_mat.sum(0)}

    X_missing = X_missing.reshape(-1, num_features)
    user_ids = np.arange(num_users).repeat(num_days)
    conditions_missing = {k: v.reshape(-1, v.shape[-1]) for k, v in condition_missing.items()}
    missing_idx = np.isnan(X_missing.sum(1))

    ## Prepare the training data with missing records
    nonzero_mean, nonzero_std = utils.zero_preserved_log_stats(X_missing)
    X_missing = utils.zero_preserved_log_normalize(X_missing, nonzero_mean, nonzero_std, log_output=LOG_SPACE, zero_id=ZERO_ID, shift=SHIFT)

    ## User encodings
    model_kwargs = USER_EMBEDDING_KWARGS["model_kwargs"]
    fit_kwargs = USER_EMBEDDING_KWARGS["fit_kwargs"]
    if ADD_USERS:        
        base_dir = f'{dataset_path}/user_encoding_models'
        model_dir = utils.find_matching_model(base_dir, model_kwargs, fit_kwargs)

        if model_dir is not None:
            entity_model = EntityEncoder.load(model_dir)
            user_gamma = np.load(f'{model_dir}/user_gamma.npy')
        else:
            entity_model = EntityEncoder(**model_kwargs)
            entity_model.fit(X_missing.reshape(num_users, num_days, -1), fit_kwargs)
            user_gamma = entity_model.transform(X_missing.reshape(num_users, num_days, -1))

            model_dir = os.path.join(base_dir, f'model_{len(os.listdir(base_dir)) + 1}')
            os.makedirs(model_dir, exist_ok=True)
            entity_model.save(model_dir)
            np.save(f'{model_dir}/user_gamma.npy', user_gamma)
        
        fig, ax = plt.subplots(figsize=(30,5))
        im = ax.imshow(user_gamma.T, aspect='auto', cmap='Purples', interpolation='nearest')
        ax.set_title('Posterior Document-Topic Dirichlet Parameters (Gamma)')
        ax.set_ylabel('Topic')
        ax.set_xlabel('Document')
        ax.set_yticks(np.arange(NUM_TOPICS))
        ax.set_xticks(np.arange(num_users))
        ax.xaxis.tick_top()
        ax.yaxis.tick_left()
        ax.xaxis.set_label_position('top')
        ax.grid(False)
        plt.yticks(np.arange(0, NUM_TOPICS, NUM_TOPICS//5))
        plt.xticks(np.arange(0, num_users, num_users//20))
        fig.colorbar(im, ax=ax)

        if not os.path.exists(f'{model_dir}/user_gamma.pdf'): fig.savefig(f'{model_dir}/user_gamma.pdf', format='pdf', bbox_inches='tight', transparent=True)
    
    ## Turn embeddings into conditions
    conditioner.add_condition(tag="users", typ="dir", support=[USER_EMBEDDING_KWARGS["fit_kwargs"]["doc_topic_prior"], entity_model.doc_lengths.max()], data=user_gamma)
    conditions_missing["users"] = user_gamma.repeat(num_days, axis=0)

    ## Continue preparing the training data
    X_missing = X_missing[~missing_idx]
    user_ids_missing = user_ids[~missing_idx]
    conditions_missing = {k: v[~missing_idx] for k, v in conditions_missing.items()}

    X_gt_condition_list["users"] = [np.array([user_gamma[user_id]]*num_missing_days) for user_id, num_missing_days in enumerate(missing_days)]

    ## split the X_missing and conditions_missing into training and validation sets
    random_idx = np.random.permutation(len(X_missing))
    val_idx = random_idx[:int(len(X_missing)*VAL_RATIO)]
    train_idx = random_idx[int(len(X_missing)*VAL_RATIO):]

    X_train, X_val = X_missing[train_idx], X_missing[val_idx]
    user_ids_train, user_ids_val = user_ids_missing[train_idx], user_ids_missing[val_idx]
    conditions_train = {k: v[train_idx] for k, v in conditions_missing.items()}
    conditions_val = {k: v[val_idx] for k, v in conditions_missing.items()}

    trainset = datasets.ConditionedDataset(inputs=X_train, conditions=conditions_train, conditioner=conditioner)
    valset = datasets.ConditionedDataset(inputs=X_val, conditions=conditions_val, conditioner=conditioner)
    print(f"Number of Training Points: {len(trainset)}")
    print(f"Number of Validation Points: {len(valset)}")

    dataset_dict = {"trainset": trainset, "valset": valset, "conditioner": conditioner, "nonzero_mean": nonzero_mean, "nonzero_std": nonzero_std, "user_gamma": user_gamma}
    return dataset_dict


def train_model(config, dataset_dict):
    model_kwargs = config["model"]
    train_kwargs = config["train"]

    model = CVAE(input_dim=dataset_dict["trainset"].inputs.shape[-1], conditioner=dataset_dict["conditioner"], **model_kwargs)
    print("Number of encoder parameters:", model.encoder._num_parameters())
    print("Number of decoder parameters:", model.decoder._num_parameters())

    trainloader = torch.utils.data.DataLoader(dataset_dict["trainset"], batch_size=train_kwargs["batch_size"], shuffle=True, drop_last=True, num_workers=1, pin_memory=True)
    valloader = torch.utils.data.DataLoader(dataset_dict["valset"], batch_size=8196, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)

    torch.cuda.empty_cache()
    model.fit(trainloader=trainloader, valloader=valloader, **train_kwargs)
    torch.cuda.empty_cache()

    model.to("cpu")
    model.prior_params = {k: v.to("cpu") for k, v in model.prior_params.items()}
    model.eval()

    return model

def main(args):
    config = load_config(args.config_path)

    np.random.seed(config["random_seed"])

    dataset_dict = prepare_data(config)

    model = train_model(config, dataset_dict)

    save_path = model.log_dir

    model_name = f'trained_model'
    model_path = f'./{save_path}/{model_name}.pt'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved at {model_path}')

    conditioner_path = f'./{save_path}/conditioner.pkl'
    with open(conditioner_path, 'wb') as f: pickle.dump(dataset_dict["conditioner"], f)
    print(f'Conditioner saved at {conditioner_path}')

    ## save the config.json file
    config_path = f'./{save_path}/config.json'
    with open(config_path, 'w') as f: json.dump(config, f)
    print(f'Config saved at {config_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a (user-informed) customisable VAE with the configuration from a JSON file.")
    parser.add_argument("--config_path", type=str, default="./config_files/config0.json", help="Path to the JSON configuration file.")
    args = parser.parse_args()
    main(args)