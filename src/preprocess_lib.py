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
    num_missing_profiles = {"user": missing_idx_mat.sum(1), "day": missing_idx_mat.sum(0)}
    X = X.reshape(-1, num_features)
    missing_idx = np.isnan(X.sum(1))

    return X, missing_idx, num_missing_profiles, missing_days

def separate_sets(data_full, condition_set, seperation_idx):
    num_users, num_days, num_features = data_full.shape
    user_ids = np.arange(num_users).repeat(num_days)

    X_sep_flat = np.reshape(data_full, (-1, num_features))[seperation_idx]
    user_ids_sep = user_ids[seperation_idx]
    condition_set_sep = {k: v[seperation_idx] for k, v in condition_set.items()}
    return X_sep_flat, user_ids_sep, condition_set_sep


def get_full_data(dataset_dir, dataset_name, resolution=1, pad=0, subsample_rate_user=1, subsample_rate_day=1):
    ## Import data
    dataset_path = os.path.join(dataset_dir, dataset_name)
    df = pd.read_csv(os.path.join(dataset_path, 'dataset.csv'))
    data, dates = df.iloc[:,:-2].values, df.date.values
    num_days, num_users = df.date.nunique(), df.user.nunique()
    print(f'Dataset: {dataset_name}')
    print(f'Loaded {len(data)} consumption profiles from {num_days} dates and {num_users} users.')

    date_dict = np.load(os.path.join(dataset_path, 'encode_dict.npy'), allow_pickle=True).item()["date_dict"]
    date_dict_inv = {v: k for k, v in date_dict.items()}
    if not os.path.exists(os.path.join(dataset_path, 'raw_dates.npy')):
        raw_dates = np.array([datetime.datetime.strptime(date_dict_inv[d], '%Y-%m-%d') for d in dates])
        np.save(os.path.join(dataset_path, 'raw_dates.npy'), raw_dates)
    else: raw_dates = np.load(os.path.join(dataset_path, 'raw_dates.npy'), allow_pickle=True)

    X = downsample_and_pad(np.reshape(data, (num_users, num_days, -1)), resolution, pad)
    X = np.reshape(data, (num_users, num_days, -1))
    X, user_mask = remove_unwanted_profiles(X)    
    X, raw_dates = subsample_data(X, np.reshape(raw_dates, (num_users, num_days))[user_mask], subsample_rate_user, subsample_rate_day)

    return X, raw_dates


def prepare_data(config_data):
    config_data["user_embedding_kwargs"]["fit_kwargs"]["lda"]["doc_topic_prior"] = 1.0/config_data["user_embedding_kwargs"]["model_kwargs"]["num_topics"]
    config_data["user_embedding_kwargs"]["fit_kwargs"]["lda"]["topic_word_prior"] = 1.0/config_data["user_embedding_kwargs"]["model_kwargs"]["num_clusters"]

    X, raw_dates = get_full_data(config_data["dataset_dir"], config_data["dataset_name"], config_data["resolution"], config_data["pad"], config_data["subsample_rate"]["user"], config_data["subsample_rate"]["day"])

    months = np.array([d.month for d in raw_dates])

    num_users, num_days = X.shape[:2]

    print("{:.<40}{:.>5}".format("Amputation Parameters", f"a={config_data['ampute_params']['a']}, b={config_data['ampute_params']['b']}"))

    X_amputed, missing_idx, _, num_missing_days = ampute_data(X, a=config_data["ampute_params"]["a"], b=config_data["ampute_params"]["b"], random_seed=config_data["random_seed"]) ## X_amputed is a flattened version of X with missing values as NaNs
    
    if config_data["random_seed"] is not None: np.random.seed(config_data["random_seed"])
    random_idx = np.random.permutation(np.setdiff1d(np.arange(X_amputed.shape[0]), np.where(missing_idx)[0]))
    num_val_data = int(random_idx.shape[0]*config_data["val_ratio"])
    num_test_data = int(random_idx.shape[0]*config_data["test_ratio"])
    num_train_data = random_idx.shape[0] - num_val_data - num_test_data

    print("{:.<40}{:.>5}".format("Number of Training Points", num_train_data))
    print("{:.<40}{:.>5}".format("Number of Testing Points", num_test_data))
    print("{:.<40}{:.>5}".format("Number of Validation Points", num_val_data))
    print("{:.<40}{:.>5}".format("Number of Missing Points", missing_idx.sum()))

    val_idx = random_idx[:num_val_data]
    test_idx = random_idx[num_val_data:num_val_data+num_test_data]
    train_idx = random_idx[num_val_data+num_test_data:]

    X_amputed_train = X_amputed.copy()
    X_amputed_train[val_idx] = np.nan
    X_amputed_train[test_idx] = np.nan

    nonzero_mean, nonzero_std = utils.zero_preserved_log_stats(X_amputed_train)
    X_amputed_train = utils.zero_preserved_log_normalize(X_amputed_train, nonzero_mean, nonzero_std, log_output=config_data["scaling"]["log_space"], zero_id=config_data["scaling"]["zero_id"], shift=config_data["scaling"]["shift"])

    condition_kwargs, condition_set = conditioning_lib.prepare_conditions(config_data["condition_tag_list"], raw_dates, data=X_amputed.reshape(num_users, num_days, -1), dataset_path=os.path.join(config_data["dataset_dir"], config_data["dataset_name"]), user_embedding_kwargs=config_data["user_embedding_kwargs"], config_dict=config_data)

    X_train, user_ids_train, conditions_train = separate_sets(X, condition_set, train_idx)
    X_train = utils.zero_preserved_log_normalize(X_train*1.0, nonzero_mean, nonzero_std, log_output=config_data["scaling"]["log_space"], zero_id=config_data["scaling"]["zero_id"], shift=config_data["scaling"]["shift"])
    months_train = months[train_idx]

    X_val, user_ids_val, conditions_val = separate_sets(X, condition_set, val_idx)
    X_val = utils.zero_preserved_log_normalize(X_val*1.0, nonzero_mean, nonzero_std, log_output=config_data["scaling"]["log_space"], zero_id=config_data["scaling"]["zero_id"], shift=config_data["scaling"]["shift"])
    months_val = months[val_idx]

    X_test, user_ids_test, conditions_test = separate_sets(X, condition_set, test_idx)
    X_test = utils.zero_preserved_log_normalize(X_test*1.0, nonzero_mean, nonzero_std, log_output=config_data["scaling"]["log_space"], zero_id=config_data["scaling"]["zero_id"], shift=config_data["scaling"]["shift"])
    months_test = months[test_idx]

    X_missing, user_ids_missing, conditions_missing = separate_sets(X, condition_set, missing_idx)
    months_missing = months[missing_idx]

    conditioner = conditioning_lib.Conditioner(**condition_kwargs, condition_set=conditions_train)
    trainset = datasets.ConditionedDataset(inputs=X_train, conditions=conditions_train, conditioner=conditioner)
    valset = datasets.ConditionedDataset(inputs=X_val, conditions=conditions_val, conditioner=conditioner)

    user_ids = {"train": user_ids_train, "val": user_ids_val, "test": user_ids_test, "missing": user_ids_missing}
    condition_set = {"train": conditions_train, "val": conditions_val, "test": conditions_test, "missing": conditions_missing}
    months = {"train": months_train, "val": months_val, "test": months_test, "missing": months_missing}

    return trainset, valset, conditioner, user_ids, months, condition_set, X_test, X_missing, num_missing_days, nonzero_mean, nonzero_std