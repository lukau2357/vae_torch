import torch

from torch import nn
from torch.nn import functional as F

from typing import List, Callable, Union, Any, TypeVar, Tuple
from config import *

class GaussianVAE(nn.Module):
    def __init__(self, sigma: float, latent_dim: int, hidden_channels : List, lr : float = 1e-4) -> None:
        super(GaussianVAE, self).__init__()
        self.latent_dim = latent_dim
        self.sigma = sigma
        self.hidden_channels = hidden_channels
        self.lr = lr

        modules = []
        in_channels = 3

        # Build Encoder
        for h_channel in hidden_channels:
            # Output shape of Conv2D is (batch_size, kernel_size, h, w)
            # Padding = (kernel_size - 1) / 2 to get same output and input size for stride = 1
            # If height is divisible by stride then the resulting height is (height / stride)
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels = h_channel,
                              kernel_size = 5, stride = 2, padding = 2),
                    nn.BatchNorm2d(h_channel),
                    nn.LeakyReLU())
            )
            in_channels = h_channel

        self.encoder = nn.Sequential(*modules)
        # Derived from previous observations
        self.last_dim = int(IMAGE_SIZE / (2 ** (len(hidden_channels))))
        self.start_size = int(hidden_channels[-1] * self.last_dim ** 2)

        self.qz_mu = nn.Linear(self.start_size, latent_dim)
        self.qz_var = nn.Linear(self.start_size, latent_dim)

        # Build Decoder, symmetrical do encoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, int(self.hidden_channels[-1] * self.last_dim ** 2))

        for i in range(-1, -len(self.hidden_channels), -1):
            # output_padding = stride - 1, padding = (kernel_size - 1) / 2 to get
            # h_out = h_in * stride. kernel_size has to be odd.
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_channels[i],
                                       hidden_channels[i - 1],
                                       kernel_size = 5,
                                       stride = 2,
                                       padding = 2,
                                       output_padding = 1),
                    nn.BatchNorm2d(hidden_channels[i - 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_channels[0],
                                               3,
                                               kernel_size = 5,
                                               stride = 2,
                                               padding = 2,
                                               output_padding = 1),
                            nn.BatchNorm2d(3),
                            nn.Sigmoid())

        self.optimizer = torch.optim.Adam(self.parameters(), lr)

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (torch.Tensor) Input torch.Tensor to encoder [N x C x H x W]
        :return: (torch.Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim = 1)

        mu = self.qz_mu(result)
        log_var = self.qz_var(result)

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (torch.Tensor) [B x D]
        :return: (torch.Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        # Reshape for transposed convolution input
        result = result.view(-1, self.hidden_channels[-1], self.last_dim, self.last_dim)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (torch.Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (torch.Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (torch.Tensor) [B x D]
        """
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, kl_weight: float = 1.0) -> List[torch.Tensor]:
        """
        Computes the VAE loss function.
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        recons_loss = torch.sum((recons - input) ** 2, dim = (1, 2, 3))
        kld_loss = 1 / (2 * self.sigma ** 2) * torch.sum(mu ** 2 + torch.exp(2 * log_var) - 1 - 2 * log_var, dim = 1)
        loss = torch.mean(recons_loss + kld_loss * kl_weight)
        return [loss, torch.mean(recons_loss).detach(), torch.mean(kld_loss).detach()]

    def sample(self, num_samples: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (torch.Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(DEVICE)
        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.Tensor) [B x C x H x W]
        :return: (torch.Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]

    def sample_supported(self, mu: torch.Tensor, logvar: torch.Tensor, num_samples: int = 5) -> torch.Tensor:
        """
        Generate samples supported by approximate posterior mean and log-variance.
        """
        eps = torch.randn(num_samples, self.latent_dim).to(DEVICE)
        z = torch.exp(logvar) * eps + mu
        return self.decode(z)