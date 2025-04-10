import json
import numpy as np
import argparse
import os, sys
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.baseline_models import QuantileRegressionNetwork
import src.preprocess_lib as preprocess_lib
from src.datasets import ContexedDataset


def load_config(json_file):
    with open(json_file, 'r') as file: return json.load(file)

def train_model(model, trainset, valset, train_kwargs, writer=None):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_kwargs["batch_size"], shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=8196, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    model.fit(trainloader=trainloader, 
                        valloader=valloader, 
                        **train_kwargs,
                        writer=writer)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    model.to("cpu")
    model.eval()
    return model

def main(config, data_config):
    trainset, valset, conditioner, _, _, _, _, condition_set, _, _, _, _, _ = preprocess_lib.prepare_data(data_config["data"])

    targets_train = trainset.inputs
    targets_val = valset.inputs
    inputs, contexts = {}, {}

    for set_type in ["train", "val"]:
        inputs[set_type] = np.concatenate([value for key, value in condition_set[set_type].items() if key.endswith("_befores")],axis=1)
        contexts[set_type] = {key: condition_set[set_type][key] for key in condition_set[set_type] if not key.endswith("_befores")}
    
    for tag in conditioner.tags.copy():
        if tag.endswith("_befores"): conditioner.tags.remove(tag)

    trainset = ContexedDataset(targets_train, inputs["train"], contexts["train"], conditioner=conditioner)
    valset = ContexedDataset(targets_val, inputs["val"], contexts["val"], conditioner=conditioner)

    model = QuantileRegressionNetwork(input_size=trainset[0][0].shape[0], 
                                        context_size=trainset[0][1].shape[0], 
                                        output_size=trainset[0][2].shape[0], 
                                        num_hidden_layers = config["model"]["num_hidden_layers"],
                                        num_neurons = config["model"]["num_neurons"],
                                        quantiles=config["model"]["quantiles"])

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if config["save_dir"] is not None:
        current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        writer = SummaryWriter(log_dir=os.path.join(config["save_dir"], config["save_tag"]+current_time))
    else: writer = SummaryWriter()

    with open(os.path.join(writer.log_dir, 'config.json'), 'w') as file: json.dump(config, file, indent=4)
    with open(os.path.join(writer.log_dir, 'data_config.json'), 'w') as file: json.dump({"data": data_config["data"]}, file, indent=4)

    model = train_model(model, trainset, valset, config["train"], writer=writer)
    model.save()
    conditioner.save(model.log_dir)
    print(f"Model saved at {model.log_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./config_files/config_baseline.json", help="Path to the JSON configuration file.")
    parser.add_argument("--data_config_path", type=str, default="./config_files/config0.json", help="Path to the JSON configuration file.")
    args = parser.parse_args()
    config = load_config(args.config_path)
    data_config = load_config(args.data_config_path)
    main(config, data_config)