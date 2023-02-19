import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
import os

from config import *
from torch_model import GaussianVAE
from PIL import Image

def transpose_image(image : torch.Tensor):
    return image.numpy().transpose((1, 2, 0))

def weighted_average(input: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    Computes the weighted average of vectors in input, assuming they are given column-wise.
    The function requires the input to be a [K x L] matrix, where K is the number of vectors,
    and L is their dimensionality. W is a vector which represents non-normalized weights for each
    of the input vectors, W >= 0.
    """
    return (W @ input) / W.sum()

def show_image(image : torch.Tensor) -> None:
    """
    Render a torch tensor as a Pillow Image.
    """
    image = image.view((3, IMAGE_SIZE, IMAGE_SIZE)).cpu()
    image = transpose_image(image)
    image = Image.fromarray((image * 255).astype(np.uint8))
    image.show()

def compare_reconstructions(image : torch.Tensor, recon : torch.Tensor, label: str) -> None:
    """
    Compares the original images with their respective VAE reconstructions.
    """
    image = image.cpu()
    recon = recon.cpu().detach()

    batch_size = image.shape[0]
    canvas = Image.fromarray(np.zeros((2 * IMAGE_SIZE, batch_size * IMAGE_SIZE)), mode = "RGB")
    
    for i in range(batch_size):
        current_image = image[i].view((3, IMAGE_SIZE, IMAGE_SIZE))
        current_image = transpose_image(current_image)
        current_image = Image.fromarray((current_image * 255).astype(np.uint8))
        canvas.paste(current_image, (IMAGE_SIZE * i, 0))
    
    for i in range(batch_size):
        current_image = recon[i].view((3, IMAGE_SIZE, IMAGE_SIZE))
        current_image = transpose_image(current_image)
        current_image = Image.fromarray((current_image * 255).astype(np.uint8))
        canvas.paste(current_image, (IMAGE_SIZE * i, IMAGE_SIZE))

    canvas.show()
    if label is not None:
        canvas.save(label)
    
def show_samples(model : GaussianVAE, label: str, num_samples = 5, per_row = 5) -> None:
    """
    Non-parameterized latent space sampling. num_samples determines the number of samples 
    to be drawn, per_row determines the number of samples to be presented in a single row 
    of the output.
    """
    model = model.to(DEVICE)
    samples = model.sample(num_samples).detach()
    samples = samples.cpu()

    nrows = int(np.ceil(num_samples / per_row))

    canvas = Image.fromarray(np.zeros((nrows * IMAGE_SIZE, IMAGE_SIZE * per_row)), mode = "RGB")
    for i in range(num_samples):
        current_sample = samples[i].view((3, IMAGE_SIZE, IMAGE_SIZE))
        current_sample = transpose_image(current_sample)
        current_sample = Image.fromarray((current_sample * 255).astype(np.uint8))
        canvas.paste(current_sample, (int(i % per_row) * IMAGE_SIZE, int(np.floor(i / per_row) * IMAGE_SIZE)))
    
    canvas.show()
    if label is not None:
        canvas.save(label)

def show_samples_baseline_weighted(model: GaussianVAE, x: torch.Tensor, W : torch.Tensor, label: str, num_samples = 1) -> None:
    """
    Sampling using a weighted baseline. x is a tensor which contains baseline samples from
    the original dataset, W is the weight vector. num_samples determines the number of samples to be derived
    using weighted baseline.
    """
    model = model.to(DEVICE)
    x = x.to(DEVICE)
    N = int(x.shape[0])

    mu, logvar = model.encode(x)
    new_mu = weighted_average(mu, W)
    new_logvar = weighted_average(logvar, W)

    samples = model.sample_supported(new_mu, new_logvar, num_samples = num_samples).detach().cpu()
    canvas = Image.fromarray(np.zeros((2 * IMAGE_SIZE, IMAGE_SIZE * N)), mode = "RGB")

    for i in range(N):
        current_image = x[i].view((3, IMAGE_SIZE, IMAGE_SIZE)).cpu()
        current_image = transpose_image(current_image)
        current_image = Image.fromarray((current_image * 255).astype(np.uint8))
        canvas.paste(current_image, (IMAGE_SIZE * i, 0))
    
    for i in range(num_samples):
        current_sample = samples[i].view((3, IMAGE_SIZE, IMAGE_SIZE)).cpu()
        current_sample = transpose_image(current_sample)
        current_sample = Image.fromarray((current_sample * 255).astype(np.uint8))
        canvas.paste(current_sample, (IMAGE_SIZE * i, IMAGE_SIZE))

    canvas.show()
    if label is not None:
        canvas.save(label)

def read_losses(target: str) -> None:
    """
    Reads loss history from a given file.
    """
    nelbo, mse, kl = [], [], []
    with open(target, "r") as f:
        reader = csv.reader(f, delimiter = ";")
        for i, row in enumerate(reader):
            if i == 0:
                continue

            nelbo.append(float(row[0]))
            mse.append(float(row[1]))
            kl.append(float(row[2]))
    
    return nelbo, mse, kl

def plot_losses(name: str) -> None:
    """
    Loss plots during training.
    """
    plt.style.use("ggplot")
    fig, ax = plt.subplots(ncols = 3)
    fig.suptitle("Losses")

    train_path = os.path.join(name, "history_train.csv")
    val_path = os.path.join(name, "history_validation.csv")
    
    nelbot, mset, klt = read_losses(train_path)
    nelbov, msev, klv = read_losses(val_path)

    ax[0].plot(nelbot, label = "train")
    ax[0].plot(nelbov, label = "validation")
    ax[0].set_title("NELBO")

    ax[1].plot(mset, label = "train")
    ax[1].plot(msev, label = "validation")
    ax[1].set_title("Reconstruction Error")

    ax[2].plot(klt, label = "train")
    ax[2].plot(klv, label = "validation")
    ax[2].set_title("KL divergence")

    # Set common legend
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc = "upper left")
    
    plt.show()