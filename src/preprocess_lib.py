import numpy as np
import pandas as pd
import os
import datetime

from . import utils, conditioning_lib, datasets

def downsample_and_pad(data, dates, resolution=1, pad=0):
    #check if pad a list of two integers
    if isinstance(pad, list): left_pad, right_pad = pad
    else: left_pad, right_pad = pad, pad
    num_features = data.shape[-1]
    
    num_left_days, num_right_days = left_pad//num_features, right_pad//num_features
    num_left_remainder, num_right_remainder = left_pad%num_features, right_pad%num_features

    if num_features%resolution != 0: raise ValueError("Resolution must divide the number of features.")
    if resolution <= 0: raise ValueError("Resolution must be positive.")
    X = np.reshape(data, (*data.shape[:-1], int(num_features/resolution), int(resolution))).sum(axis=-1)
    if pad != 0: 
        num_features = X.shape[-1]
        X_padded = X.copy()
        for left_day in range(num_left_days): X_padded = np.concatenate((np.roll(X, left_day+1, axis=1), X_padded), axis=2)
        for right_day in range(num_right_days): X_padded = np.concatenate((X_padded, np.roll(X, -right_day-1, axis=1)), axis=2)

        if num_left_remainder != 0: X_padded = np.concatenate((np.roll(X, (num_left_days+1), axis=1)[:,:,-num_left_remainder:], X_padded), axis=2)
        if num_right_remainder != 0: X_padded = np.concatenate((X_padded, np.roll(X, -(num_right_days+1), axis=1)[:,:,:num_right_remainder]), axis=2)

        if num_right_days+(num_right_remainder>0) == 0: 
            X = X_padded[:, num_left_days+(num_left_remainder>0):, :]
            dates = dates[:, num_left_days+(num_left_remainder>0):]
        else:
            X = X_padded[:, num_left_days+(num_left_remainder>0):-(num_right_days+(num_right_remainder>0)), :]
            dates = dates[:, num_left_days+(num_left_remainder>0):-(num_right_days+(num_right_remainder>0))]

    return X, dates

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

def split_datasets(num_users, num_days, missing_idx, test_ratio, val_ratio, forecasting=False):
    if forecasting:
        idx_array = np.arange(num_users*num_days).reshape(num_users, num_days)
        num_available_data = num_users*num_days - missing_idx.shape[0]

        num_trainval_days = int(num_days*(1-test_ratio))
        test_idx = np.setdiff1d(idx_array[:, np.arange(num_days)[num_trainval_days:]].flatten(), missing_idx)
        num_trainval_days = int(num_days*(1-(test_ratio**2/(test_idx.shape[0]/(num_available_data)))))
        test_idx = np.setdiff1d(idx_array[:, np.arange(num_days)[num_trainval_days:]].flatten(), missing_idx)

        random_days_idx = np.random.permutation(np.arange(num_days)[:num_trainval_days])

        num_val_days = int(num_days*val_ratio)
        val_idx = np.setdiff1d(idx_array[:, random_days_idx[:num_val_days]].flatten(), missing_idx)
        num_val_days = int(num_days*(val_ratio**2/(val_idx.shape[0]/(num_available_data))))
        val_idx = np.setdiff1d(idx_array[:, random_days_idx[:num_val_days]].flatten(), missing_idx)

        train_days_idx = random_days_idx[num_val_days:]
        train_idx = np.setdiff1d(idx_array[:, train_days_idx].flatten(), missing_idx)
    else:
        random_idx = np.random.permutation(np.setdiff1d(np.arange(num_users*num_days), missing_idx))

        val_idx = random_idx[:int(random_idx.shape[0]*val_ratio)]
        test_idx = random_idx[int(random_idx.shape[0]*val_ratio):int(random_idx.shape[0]*(val_ratio+test_ratio))]
        train_idx = random_idx[int(random_idx.shape[0]*(val_ratio+test_ratio)):]
    
    return train_idx, val_idx, test_idx

def get_full_data(dataset_dir, dataset_name, resolution=1, pad=0, subsample_rate_user=1, subsample_rate_day=1):
    ## Import data
    dataset_path = os.path.join(dataset_dir, dataset_name)
    df = pd.read_csv(os.path.join(dataset_path, 'dataset.csv'))

    if dataset_name == 'goi4_dp_full_Gipuzkoa':
        data, dates = df.iloc[:,:-2].values, df.date.values
        num_days, num_users = df.date.nunique(), df.user.nunique()
        print(f'Dataset: {dataset_name}')
        date_dict = np.load(os.path.join(dataset_path, 'encode_dict.npy'), allow_pickle=True).item()["date_dict"]
        date_dict_inv = {v: k for k, v in date_dict.items()}
        if not os.path.exists(os.path.join(dataset_path, 'raw_dates.npy')):
            raw_dates = np.array([datetime.datetime.strptime(date_dict_inv[d], '%Y-%m-%d') for d in dates])
            np.save(os.path.join(dataset_path, 'raw_dates.npy'), raw_dates)
        else: raw_dates = np.load(os.path.join(dataset_path, 'raw_dates.npy'), allow_pickle=True)
        X, raw_dates = downsample_and_pad(np.reshape(data, (num_users, num_days, -1)), np.reshape(raw_dates, (num_users, num_days)), resolution, pad)
        X, user_mask = remove_unwanted_profiles(X)
        X, raw_dates = subsample_data(X, raw_dates[user_mask], subsample_rate_user, subsample_rate_day)
        X = X.astype(float)
    elif dataset_name == 'STORM_daily':
        data, dates = df.iloc[:,:-2].values, df["Date"]
        num_days, num_users = df["Date"].nunique(), df["ID_alliander"].nunique()
        print(f'Dataset: {dataset_name}')
        raw_dates = np.array([datetime.datetime.strptime(d, '%Y-%m-%d') for d in dates])
        X, raw_dates = downsample_and_pad(np.reshape(data, (num_users, num_days, -1)), np.reshape(raw_dates, (num_users, num_days)), resolution, pad)
        X, raw_dates = subsample_data(X, raw_dates, subsample_rate_user, subsample_rate_day)
    else: raise ValueError(f"Dataset {dataset_name} not recognised. Please define the data loading procedure in preprocess_lib.py.")


    return X, raw_dates


def prepare_data(config_data):
    config_data["user_embedding_kwargs"]["fit_kwargs"]["lda"]["doc_topic_prior"] = 1.0/config_data["user_embedding_kwargs"]["model_kwargs"]["num_topics"]
    config_data["user_embedding_kwargs"]["fit_kwargs"]["lda"]["topic_word_prior"] = 1.0/config_data["user_embedding_kwargs"]["model_kwargs"]["num_clusters"]

    X, raw_dates = get_full_data(config_data["dataset_dir"], config_data["dataset_name"], config_data["resolution"], config_data["pad"], config_data["subsample_rate"]["user"], config_data["subsample_rate"]["day"])

    months = np.array([d.month for d in raw_dates])
    years = np.array([d.year for d in raw_dates])

    num_users, num_days = X.shape[:2]

    print("{:.<40}{:.>5}".format("Amputation Parameters", f"a={config_data['ampute_params']['a']}, b={config_data['ampute_params']['b']}"))

    if config_data["ampute_params"]["a"] is not None and config_data["ampute_params"]["b"] is not None:
        X_amputed, missing_mask, _, num_missing_days = ampute_data(X, a=config_data["ampute_params"]["a"], b=config_data["ampute_params"]["b"], random_seed=config_data["random_seed"]) ## X_amputed is a flattened version of X with missing values as NaNs
    else:
        X_amputed = X.reshape(-1, X.shape[-1])
        missing_mask = np.isnan(X_amputed).any(1)
        num_missing_days = {"user": np.isnan(X).any(2).sum(1), "day": np.isnan(X).any(2).sum(0)}
    
    missing_idx = np.where(missing_mask)[0]
    num_missing_data = missing_idx.shape[0]

    if config_data["random_seed"] is not None: np.random.seed(config_data["random_seed"])
    
    train_idx, val_idx, test_idx = split_datasets(num_users, num_days, missing_idx, config_data["test_ratio"], config_data["val_ratio"], forecasting=config_data["forecasting"])

    X_amputed_train = X_amputed.copy()
    X_amputed_train[val_idx] = np.nan
    X_amputed_train[test_idx] = np.nan

    if config_data["dataset_name"] == 'goi4_dp_full_Gipuzkoa':
        zero_mean, zero_std = utils.zero_preserved_log_stats(X_amputed_train)
        X_amputed_train = utils.zero_preserved_log_normalize(X_amputed_train, zero_mean, zero_std, log_output=config_data["scaling"]["log_space"], zero_id=config_data["scaling"]["zero_id"], shift=config_data["scaling"]["shift"])
        X_full_normalized = utils.zero_preserved_log_normalize(X*1.0, zero_mean, zero_std, log_output=config_data["scaling"]["log_space"], zero_id=config_data["scaling"]["zero_id"], shift=config_data["scaling"]["shift"])
        mean, std = zero_mean, zero_std
    elif config_data["dataset_name"] == 'STORM_daily':
        X_amputed_train = utils.two_sided_log_transform(X_amputed_train, alpha=config_data["scaling"]["alpha"])
        mean, std = np.nanmean(X_amputed_train, axis=0, keepdims=True), np.nanstd(X_amputed_train, axis=0, keepdims=True)
        X_amputed_train = (X_amputed_train - mean) / std
        X_full_normalized = utils.two_sided_log_normalize(X*1.0, mean, std, alpha=config_data["scaling"]["alpha"])

    condition_kwargs, condition_set = conditioning_lib.prepare_conditions(config_data["condition_tag_list"], raw_dates, data=X_full_normalized, missing_data=X_amputed_train.reshape(num_users, num_days, -1), dataset_path=os.path.join(config_data["dataset_dir"], config_data["dataset_name"]), user_embedding_kwargs=config_data["user_embedding_kwargs"], config_dict=config_data)

    missing_idx_new = missing_idx.copy()
    for _, value in condition_set.items(): 
        missing_idx_new = np.union1d(missing_idx_new, np.where(np.isnan(value).any(1))[0])

    if missing_idx_new.shape[0]>missing_idx.shape[0]:
        print(f"Adding {missing_idx_new.shape[0]-missing_idx.shape[0]} additional missing profiles due to missing condition values.")
        missing_idx = missing_idx_new
        num_missing_data = missing_idx.shape[0]
        print(f"Re-splitting datasets.")
        train_idx, val_idx, test_idx = split_datasets(num_users, num_days, missing_idx, config_data["test_ratio"], config_data["val_ratio"], forecasting=config_data["forecasting"])
    
    num_train_data, num_val_data, num_test_data = train_idx.shape[0], val_idx.shape[0], test_idx.shape[0]
    print("{:.<40}{:.>5} ({:.2f}%)".format("Number of training data", num_train_data, num_train_data / (num_users * num_days) * 100))
    print("{:.<40}{:.>5} ({:.2f}%)".format("Number of validation data", num_val_data, num_val_data / (num_users * num_days) * 100))
    print("{:.<40}{:.>5} ({:.2f}%)".format("Number of testing data", num_test_data, num_test_data / (num_users * num_days) * 100))
    print("{:.<40}{:.>5} ({:.2f}%)".format("Number of missing data", num_missing_data, num_missing_data / (num_users * num_days) * 100))

    X_train, user_ids_train, conditions_train = separate_sets(X_full_normalized, condition_set, train_idx)
    X_val, user_ids_val, conditions_val = separate_sets(X_full_normalized, condition_set, val_idx)
    X_test, user_ids_test, conditions_test = separate_sets(X_full_normalized, condition_set, test_idx)
    X_missing, user_ids_missing, conditions_missing = separate_sets(X_full_normalized, condition_set, missing_idx)

    months_train, months_val, months_test, months_missing = months[train_idx], months[val_idx], months[test_idx], months[missing_idx]
    years_train, years_val, years_test, years_missing = years[train_idx], years[val_idx], years[test_idx], years[missing_idx]

    conditioner = conditioning_lib.Conditioner(**condition_kwargs, condition_set=conditions_train)

    for i, typ in enumerate(conditioner.types):
        if typ == 'dir': 
            conditioner.transformers[conditioner.tags[i]].transform_style = config_data["dirichlet_transform_style"]
            if config_data["dirichlet_transform_style"] in ["embed"]: conditioner.cond_dim += 1

    trainset = datasets.ConditionedDataset(inputs=X_train, conditions=conditions_train, conditioner=conditioner)
    valset = datasets.ConditionedDataset(inputs=X_val, conditions=conditions_val, conditioner=conditioner)

    user_ids = {"train": user_ids_train, "val": user_ids_val, "test": user_ids_test, "missing": user_ids_missing}
    condition_set = {"train": conditions_train, "val": conditions_val, "test": conditions_test, "missing": conditions_missing}
    months = {"train": months_train, "val": months_val, "test": months_test, "missing": months_missing}
    years = {"train": years_train, "val": years_val, "test": years_test, "missing": years_missing}
    indices = {"train": train_idx, "val": val_idx, "test": test_idx, "missing": missing_idx}

    return trainset, valset, conditioner, user_ids, months, years, indices, condition_set, X_test, X_missing, num_missing_days, mean, std