import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import *

class QuantileRegressionNetwork(nn.Module):
    def __init__(self, 
                 input_size, 
                 context_size,
                 output_size,
                 num_neurons, 
                 num_hidden_layers, 
                 quantiles=[0.1, 0.5, 0.9]):
        super(QuantileRegressionNetwork, self).__init__()

        self.quantiles = quantiles        
        layers = []
        prev_size = input_size + context_size
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev_size, num_neurons))
            layers.append(nn.ReLU())
            prev_size = num_neurons

        layers.append(nn.Linear(prev_size, output_size * len(quantiles)))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs, contexts):
        x = torch.cat((inputs, contexts), dim=-1)
        x = self.network(x)
        x = x.view(x.shape[0], -1, len(self.quantiles))

        if len(self.quantiles)>1: 
            y = torch.nn.functional.softplus(x[:,:,1:], beta=1, threshold=5)
            z = torch.cat((x[:,:,:1], y), dim=-1)
            z = torch.cumsum(z, dim=-1)
            z = z.clamp(min=-5, max=5)
        return z

    def quantile_loss(self, predictions, targets):
        loss = 0.0
        for i, q in enumerate(self.quantiles):
            errors = targets - predictions[:, :, i]
            loss += torch.mean(torch.max((q - 1) * errors, q * errors)) * 1/len(self.quantiles)
        return loss

    def validate(self, valloader, device):
        self.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, contexts, targets in valloader:
                inputs, contexts, targets = inputs.to(device), contexts.to(device), targets.to(device)
                quantile_predictions = self(inputs, contexts)
                loss = self.quantile_loss(quantile_predictions, targets)
                val_loss += loss.item()

        return val_loss / len(valloader)

    def save(self, save_path=None, model_name="trained_model", device="cpu"):
        if save_path is None: save_path = self.log_dir
        model_path = os.path.join(save_path, model_name + '.pt')
        self.eval()
        self.to("cpu")
        torch.save(self.state_dict(), model_path)
        self.to(device)

    def fit(self, 
            trainloader, 
            valloader, 
            lr = 1e-3,
            weight_decay = 1e-4,
            lr_scheduling = False,
            lr_scheduling_kwargs = {"threshold":0.1, "factor":0.2, "patience":5, "min_lr":1e-5},
            epochs = 100,
            verbose_freq = 100,
            tqdm_func = tqdm,
            validation_freq = 500, 
            earlystopping = False,
            earlystopping_kwargs = {"patience":5, "delta":0.1},
            device = "cpu",
            writer = None,
            **_):
        
        if writer is None: writer = SummaryWriter()
        self.log_dir = writer.log_dir

        if tqdm_func is not None:
            total_itx_per_epoch = ((trainloader.dataset.__len__())//trainloader.batch_sampler.batch_size + (trainloader.drop_last==False))
            pbar_epx, pbar_itx = tqdm_func(total=epochs, desc="Epoch", dynamic_ncols=True), tqdm_func(total=total_itx_per_epoch, desc="Iteration in Epoch", dynamic_ncols=True)

        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        if lr_scheduling: scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", threshold_mode="abs", **lr_scheduling_kwargs)
        if earlystopping: earlystopper = EarlyStopping(**earlystopping_kwargs)

        self.to(device)
        self.train()
        
        epx, itx = 0, 0
        for _ in range(epochs):
            epx += 1
            if tqdm_func is not None: 
                pbar_epx.update(1)
                pbar_itx.reset()

            for inputs, contexts, targets in trainloader:
                itx += 1
                if tqdm_func is not None: pbar_itx.update(1)

                inputs, contexts, targets = inputs.to(device), contexts.to(device), targets.to(device)
                quantile_predictions = self.forward(inputs, contexts)
                loss = self.quantile_loss(quantile_predictions, targets)
                optim.zero_grad()
                loss.backward()
                optim.step()

                if itx%validation_freq==0 and itx>0 and valloader is not None:
                    val_loss = self.validate(valloader, device=device)
                    if tqdm_func is not None: pbar_itx.write(f"Validation -- Loss: {val_loss:.6f}")
                    else: print(f"Validation -- Loss: {val_loss:.6f}")
                    if lr_scheduling:
                        last_lr = scheduler.get_last_lr()[0]
                        scheduler.step(-val_loss)
                        if last_lr != scheduler.get_last_lr()[0]: 
                            if tqdm_func is not None: pbar_itx.write(f"Learning Rate Changed: {last_lr} -> {scheduler.get_last_lr()[0]}")
                            else: print(f"Learning Rate Changed: {last_lr} -> {scheduler.get_last_lr()[0]}")
                    if earlystopping: earlystopper(-val_loss, pbar=pbar_itx)
                    del val_loss
                    self.train()
                
                if itx%verbose_freq==0: 
                    if tqdm_func is not None: 
                        pbar_itx.write(f"Iteration {itx} -- Loss: {loss:.6f}")
                    else: print(f"Iteration {itx} -- Loss: {loss:.6f}")
                del loss

            if earlystopping and earlystopper.early_stop: break
        
        self.to("cpu")
        self.eval()