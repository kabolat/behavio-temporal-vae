import json
import argparse
import os, sys
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.vae_models import CVAE
import src.preprocess_lib as preprocess_lib


def load_config(json_file):
    with open(json_file, 'r') as file: return json.load(file)

def train_model(model, trainset, valset, train_kwargs, writer=None):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_kwargs["batch_size"], shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1024, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    model.fit(trainloader=trainloader, valloader=valloader, **train_kwargs, writer=writer)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    model.to("cpu")
    model.prior_params = {k: v.to("cpu") for k, v in model.prior_params.items()}
    model.eval()
    return model

def main(config):
    trainset, valset, conditioner, _, _, _, _, _, _, _, _, _, _ = preprocess_lib.prepare_data(config["data"])

    # trainset.num_samples = config["train"]["num_mc_samples"]
    valset.num_samples = config["train"]["validation_mc_samples"]

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
    parser.add_argument("--config_path", type=str, default="./config_files/config0_storm.json", help="Path to the JSON configuration file.")
    args = parser.parse_args()
    config = load_config(args.config_path)
    main(config)