import torch

IMAGE_SIZE = 256
BATCH_SIZE = 64
CENTER_CROP_SIZE = 148

SEED = 41
LATENT_DIM = 20
PERIOD = 10
EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"