import numpy as np
import pandas as pd
import os
import datetime

from . import utils, conditioning_lib, datasets

def downsample_and_pad(data, resolution=1, pad=0):
    num_features = data.shape[-1]

    if num_features%resolution != 0: raise ValueError("Resolution must divide the number of features.")
    if resolution <= 0: raise ValueError("Resolution must be positive.")

    X = np.reshape(data, (*data.shape[:-1], int(num_features/resolution), int(resolution))).sum(axis=-1)
    
    if pad != 0: 
        num_features = X.shape[-1]
        X = np.concatenate((X[:,:-(pad//num_features+2),-pad:], X[:,(pad//num_features+1):-(pad//num_features+1),:], X[:,(pad//num_features+2):,:pad]), axis=-1)
    return X

def remove_unwanted_profiles(data):
    num_days = data.shape[1]

    nonzero_user_mask = np.sum(np.all(data == 0, axis=2), axis=1) < num_days
    print(f'Removing {(~nonzero_user_mask).sum()} users with all-zero consumption profiles')

    positive_user_mask = np.sum(np.any(data < 0, axis=2), axis=1) == 0
    print(f'Removing {(~positive_user_mask).sum()} users with any-negative consumption profiles')

    user_mask = nonzero_user_mask & positive_user_mask
    X = data[user_mask].copy()

    return X, user_mask

def subsample_data(data, dates, user_subsample_rate=1, day_subsample_rate=1):
    X = data[::user_subsample_rate, ::day_subsample_rate]
    dates = dates[::user_subsample_rate, ::day_subsample_rate].flatten()
    return X, dates

def generate_random_enrolments(n, a=0.5, b=1.0, size=1, random_seed=None):
    if random_seed is not None: np.random.seed(random_seed)
    enrolments = np.random.binomial(n, p=np.random.beta(a, b, size=size), size=size)
    print(f"Mean of enrolments: {n*a/(a+b):.2f}")
    return enrolments

def ampute_data(data, a=0.5, b=1.0, random_seed=None):
    num_users, num_days, num_features = data.shape
    missing_days = generate_random_enrolments(n=num_days, a=a, b=b, size=num_users, random_seed=random_seed)
    
    X = data.copy().astype(float)
    for user in range(num_users): X[user, :missing_days[user]] = np.nan

    missing_idx_mat  = np.isnan(X).any(2)
    num_mising_profiles = {"user": missing_idx_mat.sum(1), "day": missing_idx_mat.sum(0)}
    X = X.reshape(-1, num_features)
    missing_idx = np.isnan(X.sum(1))

    return X, missing_idx, num_mising_profiles, missing_days

def separate_test_set(data_full, data_missing_flattened, condition_set, missing_idx, missing_days):
    num_users, num_days, num_features = data_full.shape
    user_ids = np.arange(num_users).repeat(num_days)
    X_observed = data_missing_flattened[~missing_idx]
    user_ids_observed = user_ids[~missing_idx]
    condition_set_observed = {k: v[~missing_idx] for k, v in condition_set.items()}

    X_test_flat = np.reshape(data_full, (-1, num_features))[missing_idx]
    X_test_list = [data_full[user, :missing_days[user]]*1 for user in range(num_users)]
    user_ids_test = user_ids[missing_idx]
    condition_set_test = {k: v[missing_idx] for k, v in condition_set.items()}

    return X_observed, user_ids_observed, condition_set_observed, X_test_flat, X_test_list, user_ids_test, condition_set_test

def separate_val_set(data, user_ids, condition_set, val_ratio=0.1, random_seed=None):
    if random_seed is not None: np.random.seed(random_seed)
    dataset_size = data.shape[0]
    random_idx = np.random.permutation(dataset_size)
    val_idx = random_idx[:int(dataset_size*val_ratio)]
    train_idx = random_idx[int(dataset_size*val_ratio):]

    X_train, X_val = data[train_idx], data[val_idx]
    user_ids_train, user_ids_val = user_ids[train_idx], user_ids[val_idx]
    conditions_train, conditions_val = {k: v[train_idx] for k, v in condition_set.items()}, {k: v[val_idx] for k, v in condition_set.items()}

    return X_train, user_ids_train, conditions_train, X_val, user_ids_val, conditions_val

def prepare_data(config_data):
    config_data["user_embedding_kwargs"]["fit_kwargs"]["lda"]["doc_topic_prior"] = 1.0/config_data["user_embedding_kwargs"]["model_kwargs"]["num_topics"]
    config_data["user_embedding_kwargs"]["fit_kwargs"]["lda"]["topic_word_prior"] = 1.0/config_data["user_embedding_kwargs"]["model_kwargs"]["num_clusters"]

    ## Import data
    dataset_path = os.path.join(config_data["dataset_dir"], config_data["dataset_name"])
    df = pd.read_csv(os.path.join(dataset_path, 'dataset.csv'))
    data, dates = df.iloc[:,:-2].values, df.date.values
    num_days, num_users = df.date.nunique(), df.user.nunique()
    print(f'Dataset: {config_data["dataset_name"]}')
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
    X = downsample_and_pad(np.reshape(data, (num_users, num_days, -1)), config_data["resolution"], config_data["pad"])
    X, user_mask = remove_unwanted_profiles(X)
    X, raw_dates = subsample_data(X, np.reshape(raw_dates, (num_users, num_days))[user_mask], config_data["subsample_rate"]["user"], config_data["subsample_rate"]["day"])

    num_users, num_days, num_features = X.shape

    print("{:.<40}{:.>5}".format("Number of (subsampled/filtered) users", num_users))
    print("{:.<40}{:.>5}".format("Number of (subsampled) days", num_days))
    print("{:.<40}{:.>5}".format("Number of (aggregated) features", num_features))

    X_missing, missing_idx, num_missing_profiles, missing_days = ampute_data(X, a=config_data["ampute_params"]["a"], b=config_data["ampute_params"]["b"], random_seed=config_data["random_seed"])
    
    nonzero_mean, nonzero_std = utils.zero_preserved_log_stats(X_missing)
    X_missing = utils.zero_preserved_log_normalize(X_missing, nonzero_mean, nonzero_std, log_output=config_data["scaling"]["log_space"], zero_id=config_data["scaling"]["zero_id"], shift=config_data["scaling"]["shift"])

    condition_kwargs, condition_set = conditioning_lib.prepare_conditions(config_data["condition_tag_list"], raw_dates, data=X_missing.reshape(num_users, num_days, -1), dataset_path=dataset_path, user_embedding_kwargs=config_data["user_embedding_kwargs"], config_dict=config_data)

    X_observed, user_ids_observed, condition_set_observed, X_test_flat, X_test_list, user_ids_test, condition_set_test = separate_test_set(X, X_missing, condition_set, missing_idx, missing_days)

    X_train, user_ids_train, conditions_train, X_val, user_ids_val, conditions_val = separate_val_set(X_observed, user_ids_observed, condition_set_observed, val_ratio=config_data["val_ratio"], random_seed=config_data["random_seed"])

    conditioner = conditioning_lib.Conditioner(**condition_kwargs, condition_set=condition_set_observed)
    trainset = datasets.ConditionedDataset(inputs=X_train, conditions=conditions_train, conditioner=conditioner)
    valset = datasets.ConditionedDataset(inputs=X_val, conditions=conditions_val, conditioner=conditioner)
    print(f"Number of Training Points: {len(trainset)}")
    print(f"Number of Validation Points: {len(valset)}")

    user_ids = {"train": user_ids_train, "val": user_ids_val, "test": user_ids_test}
    condition_set = {"train": conditions_train, "val": conditions_val, "test": condition_set_test}
    X_test = {"flat": X_test_flat, "list": X_test_list}

    return trainset, valset, conditioner, user_ids, condition_set, X_test