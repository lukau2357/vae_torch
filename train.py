import torch
import os
import time
import csv

from torch_model import GaussianVAE
from torch.utils.data import DataLoader
from typing import Tuple, List
from config import *

def load_latest(name : str) -> Tuple[GaussianVAE, int]:
    files = os.listdir(name)
    files = list(filter(lambda x : "csv" not in x, files))
    files = sorted(files, key = lambda x: int(x.split("_")[-1]), reverse = True)

    target = files[0]
    last_epoch = int(target.split("_")[-1])

    model = torch.load(os.path.join(name, target))
    return model.to(DEVICE), last_epoch

def average_dataset_loss(model : GaussianVAE, loader : DataLoader) -> Tuple[float, float, float]:
    nelbo_loss, recon_loss, kl_loss = torch.Tensor([0.0]).to(DEVICE), torch.Tensor([0.0]).to(DEVICE), torch.Tensor([0.0]).to(DEVICE)
    N = len(loader.dataset)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            recon, input, mu, logvar = model.forward(batch)
            nelbo_loss_b, recon_loss_b, kl_loss_b = model.loss_function(recon, input, mu, logvar)
            nelbo_loss += nelbo_loss_b
            recon_loss += recon_loss_b
            kl_loss += kl_loss_b

        # Rescale to get losses averaged over the entire dataset
        nelbo_loss *= BATCH_SIZE / N
        recon_loss *= BATCH_SIZE / N
        kl_loss *= BATCH_SIZE / N

    return nelbo_loss.item(), recon_loss.item(), kl_loss.item()

def writerow(target: str, row_data: List) -> None:
    with open(target, "a") as f:
        writer = csv.writer(f, delimiter = ";", lineterminator = "\n")
        writer.writerow(row_data)

def train(model : GaussianVAE, name : str, train_loader : DataLoader, validation_loader : DataLoader) -> None:
    if not os.path.exists(name):
        os.mkdir(name)

    history_train_path = os.path.join(name, "history_train.csv")
    history_validation_path = os.path.join(name, "history_validation.csv")

    if not os.path.exists(history_train_path):
        writerow(history_train_path, ["nelbo_loss", "reconstruction_loss", "kl_divergence_loss"])

    if not os.path.exists(history_validation_path):
        writerow(history_validation_path, ["nelbo_loss", "reconstruction_loss", "kl_divergence_loss"])
    
    last_epoch = 0
    if model is None:
        model, last_epoch = load_latest(name)
    
    for i in range(last_epoch + 1, EPOCHS + 1):
        start = time.time()
        print("Started epoch: {:d}".format(i))
        
        for batch in train_loader:
            # Reset parameter derivatives from previous iteration
            model.optimizer.zero_grad()
            # Batches sent to the main compute device, as as the model.
            batch = batch.to(DEVICE)
            recon, input, mu, logvar = model.forward(batch)

            negative_elbo, recon_loss, kl = model.loss_function(recon, input, mu, logvar, kl_weight = 0.1)
            
            # Update parameters
            negative_elbo.backward()
            model.optimizer.step()

        end = time.time()
        print("Epoch {:d} ended in {:.4f} minutes".format(i, (end - start) / 60))
        
        # Loss computations on train and validation
        nelbo_train, recon_train, kl_train = average_dataset_loss(model, train_loader)
        nelbo_val, recon_val, kl_val = average_dataset_loss(model, validation_loader)

        print("Training dataset losses:")
        print("NELBO: {:.4f} \t Reconstruction Loss: {:.4f} \t KL Loss: {:.4f}".format(nelbo_train, recon_train, kl_train))
        print("Validation dataset losses:")
        print("NELBO: {:.4f} \t Reconstruction Loss: {:.4f} \t KL Loss: {:.4f}".format(nelbo_val, recon_val, kl_val))

        # Write losses to history files
        writerow(history_train_path, [nelbo_train, recon_train, kl_train])
        writerow(history_validation_path, [nelbo_val, recon_val, kl_val])

        # Saving the model
        if i % PERIOD == 0:
            torch.save(model, os.path.join(name, "{}_{:d}".format(name, i)))