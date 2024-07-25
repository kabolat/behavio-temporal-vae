import numpy as np

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