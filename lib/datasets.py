import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from itertools import product
from .utils import *

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

class Conditioner():
    def __init__(self, tags, supports, types):
        self.tags = tags
        self.supports = supports
        self.types = types
        self.cond_dim = 0
        self.init_transformers()
    
    def init_transformers(self):
        self.transformers = {}
        for tag, support, typ in zip(self.tags, self.supports, self.types):
            if typ == "circ":
                self.transformers[tag] = CircularTransformer(max_conds=np.max(support), min_conds=np.min(support))
                self.cond_dim += 2
            elif typ == "cat":
                raise NotImplementedError("Categorical transformer is not implemented yet.")
            elif typ == "cont":
                raise NotImplementedError("Continuous transformer is not implemented yet.")
            else:
                raise ValueError("Unknown type.")
    
    def transform(self, data):
        transformed_data = []
        for tag, typ in zip(self.tags, self.types):
            if typ == "circ":
                transformed_data.append(self.transformers[tag].transform(data[tag][:, np.newaxis]))
            elif typ == "cat":
                raise NotImplementedError("Categorical transformer is not implemented yet.")
            elif typ == "cont":
                raise NotImplementedError("Continuous transformer is not implemented yet.")
            else:
                raise ValueError("Unknown type.")
        condition_set = np.concatenate(transformed_data, axis=1)
        return condition_set
    
    def get_random_conditions(self, num_samples=1, random_seed=0):
        np.random.seed(random_seed)
        random_conditions = {}
        for tag, typ in zip(self.tags, self.types):
            if typ == "circ":
                random_conditions[tag] = np.random.randint(self.transformers[tag].min_conds, self.transformers[tag].max_conds+1, num_samples)
            elif typ == "cat":
                raise NotImplementedError("Categorical transformer is not implemented yet.")
            elif typ == "cont":
                raise NotImplementedError("Continuous transformer is not implemented yet.")
            else:
                raise ValueError("Unknown type.")
        condition_set = self.transform(random_conditions)
        return condition_set, random_conditions
    
    def get_all_conditions(self):
        all_conditions = {}
        for tag, typ in zip(self.tags, self.types):
            if typ == "circ":
                all_conditions[tag] = np.arange(self.transformers[tag].min_conds, self.transformers[tag].max_conds+1)
            elif typ == "cat":
                raise NotImplementedError("Categorical transformer is not implemented yet.")
            elif typ == "cont":
                raise NotImplementedError("Continuous transformer is not implemented yet.")
            else:
                raise ValueError("Unknown type.")
        ## product the conditions
        all_conditions = np.array(list(product(*all_conditions.values())))
        all_conditions = {tag: all_conditions[:, i] for i, tag in enumerate(self.tags)}
        condition_set = self.transform(all_conditions)
        return condition_set, all_conditions

class ConditionedDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, conditions=None):
        self.inputs = inputs
        self.conditions = conditions
        
    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        if self.conditions is None: return self.inputs[idx], torch.tensor([])
        else: return self.inputs[idx], self.conditions[idx]

class UserDayDataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        """
        inputs: (num_users, num_days, num_features)
        conditions: (num_users, num_days, num_conditions)
        """
        self.num_users, self.num_days, self.num_features = inputs.shape
        self.flatten_inputs = inputs.flatten(end_dim=1)  # (num_users*num_days, num_features)
        user_list = torch.arange(0,self.num_users).repeat_interleave(self.num_days)
        day_list = torch.arange(0,self.num_days).repeat(self.num_users)
        self.user_day_list = torch.stack([user_list, day_list], dim=1)

    def __len__(self):
        return self.num_users*self.num_days

    def __getitem__(self, idx):
        indicator_dict = {"idx":idx, "user_day":self.user_day_list[idx]}
        return self.flatten_inputs[idx], torch.tensor([]), indicator_dict

class MultiIndexSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, user_batch_size, day_batch_size):
        self.num_users, self.num_days = dataset.num_users, dataset.num_days
        assert user_batch_size <= self.num_users, "user_batch_size must be smaller than the number of users"
        assert day_batch_size <= self.num_days, "day_batch_size must be smaller than the number of days"

        self.batch_size = user_batch_size*day_batch_size
        self.user_batch_size = user_batch_size
        self.day_batch_size = day_batch_size

        self.idx_mat = torch.arange(0, self.num_users*self.num_days).reshape(self.num_users, self.num_days)

    def __iter__(self):
        permed_users = torch.randperm(self.num_users)
        chunked_users = self.equal_chunk(permed_users, self.user_batch_size)
        for user_chunk in chunked_users:
            permed_days = torch.randperm(self.num_days)
            chunked_days = self.equal_chunk(permed_days, self.day_batch_size)
            for day_chunk in chunked_days:
                yield self.idx_mat[user_chunk][:,day_chunk].flatten()
    
    def equal_chunk(self, indices, chunk_size):
        chunks = torch.split(indices, chunk_size)
        if chunks[-1].shape[0] < chunk_size:
            chunks = chunks[:-1]
        return chunks

    def __len__(self):
        return self.num_users*self.num_days
















class ModifiedDataset(Dataset):
    def prepareData(self, data_folder, train, input_idx=[], cond_idx=[]):
        if train:
            #region Training data
            csv_file = data_folder+"/trainset.csv"
            data = pd.read_csv(csv_file,sep=',',header=None).values
            data = torch.tensor(data).float()

            self.inputs = data[:,input_idx]
            self.conditions = circlize(data[:,cond_idx])
            
            self.mean = torch.zeros(input_idx.__len__())
            self.std = torch.ones(input_idx.__len__())
            
            if self.normalize:
                self.mean = self.inputs.mean(dim=0)
                self.inputs -= self.mean
                
                self.std = self.inputs.std(dim=0)
                self.inputs /= self.std

            #endregion
            transform_dict = {"mean": self.mean, "std": self.std}
            torch.save(transform_dict, data_folder+"/transform_dict.pt")
            
            #region Validation data
            csv_file = data_folder+"/valset.csv"
            valdata = pd.read_csv(csv_file,sep=',',header=None).values
            valdata = torch.tensor(valdata).float()
            
            self.val_inputs = valdata[:,input_idx]
            self.val_conditions = circlize(valdata[:,cond_idx])

            if self.normalize: self.val_inputs = normalize(self.val_inputs, self.mean, self.std)
            #endregion
            
        else:
            csv_file = data_folder+"/testset.csv"
            testdata = pd.read_csv(csv_file,sep=',',header=None).values
            testdata = torch.tensor(testdata).float()
            
            tr_dict = torch.load(data_folder+"/transform_dict.pt")
            self.mean = tr_dict["mean"]
            self.std = tr_dict["std"]
            
            self.inputs = testdata[:,input_idx]
            if self.normalize: self.inputs = (self.inputs - self.mean)/self.std
            
            self.int_conditions = testdata[:,cond_idx]
            self.conditions = circlize(self.int_conditions)

    def splitDataset(self,data_folder, file_name, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
        from sklearn.model_selection import train_test_split
        
        df = pd.read_csv(data_folder+file_name, sep=',', low_memory=True)
        train_val, test = train_test_split(df, train_size=train_ratio+val_ratio)
        train, val = train_test_split(train_val, train_size=train_ratio/(train_ratio+val_ratio))
        train.to_csv(data_folder+"/trainset.csv", index=False, header=False)
        val.to_csv(data_folder+"/valset.csv", index=False, header=False)
        test.to_csv(data_folder+"/testset.csv", index=False, header=False)

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.inputs[idx].to(self.device), self.conditions[idx].to(self.device)

class LCLDataset(ModifiedDataset):
    input_dim = 48
    cond_dim = 2*2          ## Assuming all of them are circular (cos-sin)
    ##TODO: Categorical (one-hot encoded) condition selection

    def __init__(self, data_folder="../../DATA/lcl", train=True, device="cpu", normalize=True, file_name = "/daily_data.csv", **_):
        ##TODO: Default data folder is so spesific. Use global variable for data folder
        
        self.device = device
        self.normalize = normalize
        input_idx = [*range(5,5+LCLDataset.input_dim)]
        cond_idx = [[2], [4]]
        if not os.path.isfile(data_folder+"/valset.csv"):
            print("The dataset required splitting!")
            self.splitDataset(data_folder, file_name)
        self.prepareData(data_folder=data_folder, train=train, input_idx=input_idx, cond_idx=cond_idx) ##TODO: Avoid keyword replication

class LCLSmallDataset(LCLDataset):
    input_dim = 48
    cond_dim = 2*2

    def __init__(self, data_folder="../../DATA/lcl_small", train=True, device="cpu", normalize=True, file_name = "/daily_data.csv"):
        ##TODO: Default data folder is so spesific. Use global variable for data folder
        keys = locals()
        del keys['self']
        super().__init__(**keys)

class LCLXSmallDataset(LCLDataset):
    input_dim = 48
    cond_dim = 2*2

    def __init__(self, data_folder="../../DATA/lcl_xsmall", train=True, device="cpu", normalize=True, file_name = "/daily_data.csv"):
        ##TODO: Default data folder is so spesific. Use global variable for data folder
        keys = locals()
        del keys['self']
        super().__init__(**keys)

class IndividualHouseholdDataset(ModifiedDataset):
    input_dim = 48
    cond_dim = 2*2

    def __init__(self, data_folder="../../DATA/individual", train=True, device="cpu", normalize=True, file_name = "/daily_data.csv", **_):
        self.device = device
        self.normalize = normalize
        input_idx = [*range(5,5+IndividualHouseholdDataset.input_dim)]
        cond_idx = [[2], [4]]
        if not os.path.isfile(data_folder+"/valset.csv"):
            print("The dataset required splitting!")
            self.splitDataset(data_folder, file_name)
        self.prepareData(data_folder=data_folder, train=train, input_idx=input_idx, cond_idx=cond_idx)

class ElectricityDataset(ModifiedDataset):
    input_dim = 48
    cond_dim = 2*2

    def __init__(self, data_folder="../../DATA/electricity", train=True, device="cpu", normalize=True, file_name = "/daily_data.csv", **_):
        self.device = device
        self.normalize = normalize
        input_idx = [*range(5,5+ElectricityDataset.input_dim)]
        cond_idx = [[2], [4]]
        if not os.path.isfile(data_folder+"/valset.csv"):
            print("The dataset required splitting!")
            self.splitDataset(data_folder, file_name)
        self.prepareData(data_folder=data_folder, train=train, input_idx=input_idx, cond_idx=cond_idx) 

def return_data(dset_name, batch_size=64, train=True, device="cpu", normalize=True, shuffle=True):

    dset_keys = {'train':train, 'device':device, 'normalize':normalize}

    if dset_name == "lcl":
        dset = LCLDataset(**dset_keys)
    elif dset_name == "lcl_small":
        dset = LCLSmallDataset(**dset_keys)
    elif dset_name == "lcl_xsmall":
        dset = LCLXSmallDataset(**dset_keys)
    elif dset_name == "indiv":
        dset = IndividualHouseholdDataset(**dset_keys)
    elif dset_name == "electricity":
        dset = ElectricityDataset(**dset_keys)
    else:
        raise NotImplementedError
    
    if train:
        train_loader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        return train_loader, {"inputs":dset.val_inputs.to(device), "conditions":dset.val_conditions.to(device)}, {"mean":dset.mean.to(device), "std":dset.std.to(device)} 
    else:
        return dset
    
def return_dim(dset_name):
    if dset_name == "lcl":
        return LCLDataset.input_dim, LCLDataset.cond_dim 
    elif dset_name == "lcl_small":
        return LCLSmallDataset.input_dim, LCLSmallDataset.cond_dim
    elif dset_name == "lcl_xsmall":
        return LCLXSmallDataset.input_dim, LCLXSmallDataset.cond_dim
    elif dset_name == "indiv":
        return IndividualHouseholdDataset.input_dim, IndividualHouseholdDataset.cond_dim 
    elif dset_name == "electricity":
        return ElectricityDataset.input_dim, ElectricityDataset.cond_dim 
    else:
        raise NotImplementedError

if __name__ == "__main__":

    time_slots = get_timeslots(delta=30)
    d = 10

    import matplotlib.pyplot as plt
    from mlxtend.plotting import scatterplotmatrix
    train_loader, valset, stats = return_data(dset_name="lcl", normalize=False)
    scatterplotmatrix(valset["inputs"][:,::d].numpy(), s=0.5, alpha=0.7, names=time_slots[::d])
    plt.tight_layout()

    train_loader, valset, stats = return_data(dset_name="lcl")
    scatterplotmatrix(valset["inputs"][:,::d].numpy(), s=0.5, alpha=0.7, names=time_slots[::d])
    plt.tight_layout()
    
    train_loader, valset, stats = return_data(dset_name="lcl")
    scatterplotmatrix(valset["inputs"][:,::d].numpy(), s=0.5, alpha=0.7, names=time_slots[::d])
    plt.tight_layout()
    
    plt.show()
    pass
    

