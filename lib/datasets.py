import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import datetime
from itertools import product
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from .utils import *

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

class Conditioner():
    def __init__(self, tags, supports, types, condition_set=None):
        self.tags = tags
        self.supports = supports
        self.types = types
        self.cond_dim = 0
        self.transformers = {}
        self.init_transformers(condition_set)

    def add_transformer(self, tag, support, typ, data=None):
        if typ == "circ":
            self.transformers[tag] = CircularTransformer(max_conds=np.max(support), min_conds=np.min(support))
            self.cond_dim += 2
        elif typ == "cat":
            self.transformers[tag] = OneHotEncoder(sparse_output=False).fit(data)
            self.cond_dim += self.transformers[tag].categories_[0].shape[0]
        elif typ == "cont":
            self.transformers[tag] = MinMaxTransformer(feature_range=(-1, 1)).fit(data)
            self.cond_dim += 1
        elif typ == "ord":
            ## always give the ascending support!
            self.transformers[tag] = OrdinalEncoder(categories=[support]).fit(data)
            self.cond_dim += 1
        elif typ == "dir":
            num_dims = data.shape[1]
            self.transformers[tag] = DirichletTransformer(num_dims=num_dims, transform_style="sample")
            self.cond_dim += num_dims
        else:
            raise ValueError("Unknown type.")
    
    def init_transformers(self, data):
        for tag, support, typ in zip(self.tags, self.supports, self.types):
            self.add_transformer(tag, support, typ, data[tag])
    
    def add_condition(self, tag, support, typ, data=None):
        self.tags.append(tag)
        self.supports.append(support)
        self.types.append(typ)
        self.add_transformer(tag, support, typ, data)
    
    def transform(self, data):
        transformed_data = []
        for tag in self.tags: 
            data_ = data[tag]
            transformed_data.append(self.transformers[tag].transform(data_))
        return np.concatenate(transformed_data, axis=1)
    
    def get_random_conditions(self, num_samples=1, random_seed=None):
        if random_seed is not None: np.random.seed(random_seed)
        random_conditions = {}
        for tag, typ, support in zip(self.tags, self.types, self.supports):
            if typ == "circ":
                random_conditions[tag] = np.random.randint(self.transformers[tag].min_conds, self.transformers[tag].max_conds+1, num_samples)[...,None]
            elif typ == "cat" or typ == "ord":
                random_conditions[tag] = np.random.choice(self.transformers[tag].categories_[0], num_samples)[...,None]
            elif typ == "cont":
                random_conditions[tag] = self.transformers[tag].inverse_transform(np.random.rand(num_samples)[:, np.newaxis]).squeeze(-1)[...,None]
            elif typ == "dir":
                rnd_doc_length = np.random.uniform(low=support[0], high=support[1], size=num_samples)
                random_conditions[tag] = np.random.dirichlet(alpha=[1.0]*self.transformers[tag].num_dims, size=num_samples)*rnd_doc_length[:, np.newaxis]
            else:
                raise ValueError("Unknown type.")
        condition_set = self.transform(random_conditions)
        return condition_set, random_conditions

class ConditionedDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, conditions=None, conditioner=None):
        self.inputs = inputs
        self.conditions = conditions ##raw
        self.conditioner = conditioner

    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        input_ = self.inputs[idx]
        condition_ = self.conditioner.transform({key: conditon[[idx]] for key, conditon in self.conditions.items()}).squeeze()
        return torch.tensor(input_).float(), torch.tensor(condition_).float()


class GOI4Dataset():
    def __init__(self, dataset_name, data_folder_path, dataset_kwargs, random_seed=0):
        self.dataset_name = dataset_name
        self.data_folder_path = data_folder_path
        self.random_seed = random_seed
        self.dataset_kwargs = dataset_kwargs

        X, dates, users = self.get_raw_data()
        conditoner, condition_set = self.get_conditions(dates)
        X, condition_set = self.set_resolution(X, condition_set, resolution=self.dataset_kwargs["resolution"])
        X, condition_set = self.clean_data(X, condition_set)
        X_missing, condition_missing, missing_days = self.ampute_dataset(X, condition_set)
        X, X_missing, condition_set, condition_missing, missing_idx, missing_num_labels, X_gt_list, X_gt_condition_list = self.subsample_dataset(X, X_missing, condition_set, condition_missing, missing_days, user_subsample_rate=self.dataset_kwargs["subsample_rate"]["user"], day_subsample_rate=self.dataset_kwargs["subsample_rate"]["day"])
        X_missing, nonzero_mean, nonzero_std = self.transform_data(X_missing, shift=self.dataset_kwargs["transform"]["shift"], zero_id=self.dataset_kwargs["transform"]["zero_id"], log_space=self.dataset_kwargs["transform"]["log_space"])

        self.data = X
        self.data_missing = X_missing
        self.condition_set = condition_set
        self.condition_missing = condition_missing
        self.missing_idx = missing_idx
        self.missing_num_labels = missing_num_labels
        self.X_gt_list = X_gt_list
        self.X_gt_condition_list = X_gt_condition_list

        self.nonzero_mean = nonzero_mean
        self.nonzero_std = nonzero_std


    def get_raw_data(self):
        df = pd.read_csv(f'{self.data_folder_path}/{self.dataset_name}/dataset.csv')

        data, dates, users = df.iloc[:,:-2].values, df.date.values, df.user.values

        date_ids, user_ids = df.date.unique(), df.user.unique()
        self.num_days, self.num_users = len(date_ids), len(user_ids)
        print(f'Loaded {len(data)} consumption profiles from {self.num_days} dates and {self.num_users} users')

        self.metadata = pd.read_csv(f'{self.data_folder_path}/{self.dataset_name}/metadata.csv')

        return data, dates, users
    
    def get_conditions(self, dates):
        self.date_dict = np.load(f'{self.data_folder_path}/{self.dataset_name}/encode_dict.npy', allow_pickle=True).item()["date_dict"]
        self.date_dict_inv = {v: k for k, v in self.date_dict.items()}
        if not os.path.exists(f'{self.data_folder_path}/{self.dataset_name}/raw_dates.npy'):
            raw_dates = np.array([datetime.datetime.strptime(self.date_dict_inv[d], '%Y-%m-%d') for d in dates])
            np.save(f'{self.data_folder_path}/{self.dataset_name}/raw_dates.npy', raw_dates)
        else: raw_dates = np.load(f'{self.data_folder_path}/{self.dataset_name}/raw_dates.npy', allow_pickle=True)
        
        months = np.array([d.month for d in raw_dates])
        weekdays = np.array([d.weekday() for d in raw_dates])
        is_weekend = np.array([int(d.weekday() >= 5) for d in raw_dates])

        df_temp = pd.read_csv(f'{self.data_folder_path}/{self.dataset_name}/spain_temp_daily.csv')
        df_temp.index = pd.to_datetime(df_temp['date'])
        df_temp.drop(columns='date', inplace=True)
        df_temp = df_temp.loc[raw_dates]

        df_prec = pd.read_csv(f'{self.data_folder_path}/{self.dataset_name}/spain_prec_daily.csv')
        df_prec.index = pd.to_datetime(df_prec['date'])
        df_prec.drop(columns='date', inplace=True)
        df_prec = df_prec.loc[raw_dates]
        df_prec = df_prec.sort_values(by='prec_total')

        condition_kwargs = {}

        condition_kwargs["tags"], condition_kwargs["types"], condition_kwargs["supports"], condition_set  = [], [], [], {}

        if self.dataset_kwargs["conditions"]["add_months"]: 
            condition_kwargs["tags"].append("months")
            condition_kwargs["types"].append("circ")
            condition_kwargs["supports"].append(np.unique(months).tolist())
            condition_set["months"] = months[...,None]
        if self.dataset_kwargs["conditions"]["add_weekdays"]:
            condition_kwargs["tags"].append("weekdays")
            condition_kwargs["types"].append("circ")
            condition_kwargs["supports"].append(np.unique(weekdays).tolist())
            condition_set["weekdays"] = weekdays[...,None]
        if self.dataset_kwargs["conditions"]["add_is_weekend"]:
            condition_kwargs["tags"].append("is_weekend")
            condition_kwargs["types"].append("cat")
            condition_kwargs["supports"].append([0, 1])
            condition_set["is_weekend"] = is_weekend[...,None]
        if self.dataset_kwargs["conditions"]["add_temp_min"]:
            condition_kwargs["tags"].append("temp_min")
            condition_kwargs["types"].append("cont")
            condition_kwargs["supports"].append([df_temp[condition_kwargs["tags"][-1]].min(), df_temp[condition_kwargs["tags"][-1]].max()])
            condition_set["temp_min"] = df_temp[condition_kwargs["tags"][-1]].values[...,None]
        if self.dataset_kwargs["conditions"]["add_temp_max"]:
            condition_kwargs["tags"].append("temp_max_delta")
            condition_kwargs["types"].append("cont")
            condition_kwargs["supports"].append([df_temp[condition_kwargs["tags"][-1]].min(), df_temp[condition_kwargs["tags"][-1]].max()])
            condition_set["temp_max_delta"] = df_temp[condition_kwargs["tags"][-1]].values[...,None]
        if self.dataset_kwargs["conditions"]["add_precip_level"]:
            condition_kwargs["tags"].append("precipitation_level")
            condition_kwargs["types"].append("ord")
            condition_kwargs["supports"].append(np.unique(df_prec["label"]).tolist())
            condition_set["precipitation_level"] = df_prec["label"].values[...,None]
        
        conditioner = Conditioner(**condition_kwargs, condition_set=condition_set)

        return conditioner, condition_set

    def set_resolution(self, X, condition_set, resolution):
        RESOLUTION = 1 #in hours

        if RESOLUTION == 12:
            X = np.reshape(X, (-1, 24))
            X = np.reshape(np.concatenate([X[:,6:], X[:,:6]], axis=-1), (self.num_users, self.num_days, int(24/RESOLUTION), int(RESOLUTION))).sum(axis=-1)    #circle shift the last dimension of X
        else: X = np.reshape(X, (self.num_users, self.num_days, int(24/RESOLUTION), int(RESOLUTION))).sum(axis=-1)

        condition_set = {k: np.reshape(v, (self.num_users, self.num_days, -1)) for k, v in condition_set.items()}

        return X, condition_set
    
    def clean_data(self, X, condition_set):
        nonzero_user_mask = np.sum(np.all(X == 0, axis=2), axis=1) < self.num_days
        print(f'Removing {(~nonzero_user_mask).sum()} users with all-zero consumption profiles')
        positive_user_mask = np.sum(np.any(X < 0, axis=2), axis=1) == 0
        print(f'Removing {(~positive_user_mask).sum()} users with any-negative consumption profiles')
        user_mask = nonzero_user_mask & positive_user_mask
        X = X[user_mask]
        condition_set = {k: v[user_mask] for k, v in condition_set.items()}
        
        return X, condition_set

    def ampute_dataset(self, X, condition_set):
        np.random.seed(self.random_seed)
        n, a, b = self.num_days, 0.85, 10.0
        missing_days = np.random.binomial(n, p=np.random.beta(a, b, size=X.shape[0]), size=X.shape[0])
        print(f"Mean of missing days: {n*a/(a+b):.2f}")

        X_missing = X.copy().astype(float)
        condition_missing = {k: v.copy().astype(float) for k, v in condition_set.items()}

        for user in range(X.shape[0]): 
            X_missing[user, :missing_days[user]] = np.nan
            for k in condition_missing.keys():
                condition_missing[k][user, :missing_days[user]] = np.nan
        
        return X_missing, condition_missing, missing_days

    def subsample_dataset(self, X, X_missing, condition_set, condition_missing, missing_days, user_subsample_rate=1, day_subsample_rate=1):
        X, X_missing = X[::user_subsample_rate, ::day_subsample_rate, :], X_missing[::user_subsample_rate, ::day_subsample_rate, :]
        condition_set = {k: v[::user_subsample_rate, ::day_subsample_rate, :] for k, v in condition_set.items()}
        condition_missing = {k: v[::user_subsample_rate, ::day_subsample_rate, :] for k, v in condition_missing.items()}
        self.num_users, self.num_days, self.num_features = X.shape
        X_gt_list = [X[user, :missing_days[user]]*1 for user in range(self.num_users)]
        X_gt_condition_list = {k: [v[user, :missing_days[user]]*1 for user in range(self.num_users)] for k, v in condition_set.items()}

        print("{:.<40}{:.>5}".format("Number of (subsampled/filtered) users", self.num_users))
        print("{:.<40}{:.>5}".format("Number of (subsampled) days", self.num_days))
        print("{:.<40}{:.>5}".format("Number of (aggregated) features", self.num_days))

        missing_idx_mat  = np.isnan(X_missing).any(2)
        missing_num_labels = {"user": missing_idx_mat.sum(1), "day": missing_idx_mat.sum(0) }

        X_missing = X_missing.reshape(-1, self.num_features)
        conditions_missing = {k: v.reshape(-1, v.shape[-1]) for k, v in condition_missing.items()}
        missing_idx = np.isnan(X_missing.sum(1))

        return X, X_missing, condition_set, conditions_missing, missing_idx, missing_num_labels, X_gt_list, X_gt_condition_list
    
    def transform_data(self, X_missing, shift=1, zero_id=-3, log_space=True):
            nonzero_mean, nonzero_std = zero_preserved_log_stats(X_missing)
            X_missing = zero_preserved_log_normalize(X_missing, nonzero_mean, nonzero_std, log_output=log_space, zero_id=zero_id, shift=shift)

            return X_missing, nonzero_mean, nonzero_std