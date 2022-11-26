import torch
import numpy as np
import matplotlib.pyplot as plt

from torchsummary import summary
from torch_model import GaussianVAE
from torch_dataset import *
from config import *
from PIL import Image
from train import train, load_latest
from utilities import *

# Comment to get variant samples from latebt space
torch.manual_seed(SEED)

if __name__ == "__main__":
     # Instantiating a GaussianVAE model.
     # model = GaussianVAE(1, LATENT_DIM, [32, 64, 128, 256, 512]).to(DEVICE)
     # summary(model, (3, IMAGE_SIZE, IMAGE_SIZE), 64)
 
     train_loader = get_data_loader("celeba_train", alpha = 1)
     validation_loader = get_data_loader("celeba_validation", alpha = 1)

     # Initiate model training
     # train(None, "full_aug_const_diminish_20_0_1", train_loader, validation_loader)

     # Load a pretrained model
     model, _ = load_latest("full_aug_const_diminish_20_0_1")
     # Model summary   
     # summary(model, (3, IMAGE_SIZE, IMAGE_SIZE), 1)
     
     # Loading N images from the training and validation datasetse for demo purposes.
     N = 5
     image_val = next(iter(validation_loader))[:N].to(DEVICE)
     image_train = next(iter(train_loader))[:N].to(DEVICE)

     # Possible to target specific images
     # image_val = get_images("celeba_validation", [162789, 162793, 162890, 162953, 162969])

     # Comparing original images with their reconstructions, without saving.
     '''
     recon_train = model.generate(image_train)
     recon_val = model.generate(image_val)
     compare_reconstructions(image_train, recon_train, None)
     compare_reconstructions(image_val, recon_val, None)
     '''

     # Losses plot
     # plot_losses("full_aug_const_diminish_20_0_1")
     
     # Unsupported sampling
     # show_samples(model, None, num_samples = 15)

     # Supported sampling, taking images from the validation dataset for example.
     W = torch.Tensor(np.repeat(1, N)).to(DEVICE)
     show_samples_baseline_weighted(model, image_val, W, None, num_samples = 1)