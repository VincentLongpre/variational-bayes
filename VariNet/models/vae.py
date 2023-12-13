import torch
from torch import nn
import numpy as np
from ..utils.dists import DiagonalGaussian

class VAE(nn.Module):
  """
  Variational Autoencoder (VAE) for generative modeling and latent space representation.
  """
  def __init__(self, feature_size, encoder, decoder, z_dim=100, device=torch.device("cuda:0")):
    super(VAE, self).__init__()

    self.z_dim = z_dim
    self.device = device
    self.encoder = encoder
    self.decoder = decoder
    self.mean = nn.Linear(feature_size, z_dim)
    self.logvar = nn.Linear(feature_size, z_dim)

  def encode(self, x):
    """
    Returns the posterior distribution q_\phi(z | x)
    """
    x = self.encoder(x)
    mean = self.mean(x)
    logvar = self.logvar(x)
    posterior = DiagonalGaussian(mean,logvar,device=self.device)
    return posterior

  def decode(self, z):
    """
    Returns the conditional distribution (likelihood) p_\theta(x | z)
    """
    mean = self.decoder(z)
    conditional = DiagonalGaussian(mean,device=self.device)
    return conditional

  def sample(self, batch_size):
    """
    Returns samples generated samples using the decoder
    """
    z = torch.randn((batch_size,self.z_dim)).to(self.device)
    conditional = self.decode(z)
    mode = conditional.mode()
    return mode

  def log_likelihood(self, x, K=100):
    """
    Approximate the log-likelihood of the data using Importance Sampling
    Inputs:
      x: Data sample tensor
      K: Number of samples to use to approximate p_\theta(x)
    Returnss log likelihood of the sample x in the VAE model using K samples
    """
    posterior = self.encode(x)
    prior = DiagonalGaussian(torch.zeros_like(posterior.mean),device=self.device)

    log_likelihood = torch.zeros(x.shape[0], K).to(self.device)
    for i in range(K):
      z = posterior.sample()
      recon = self.decode(z)
      log_likelihood[:, i] = - recon.nll(x) - prior.nll(z) + posterior.nll(z)
      del z, recon

    ll = torch.logsumexp(log_likelihood,1) - np.log(K)
    return ll

  def forward(self, x):
    """
    Input:
      x
    Returns:
      reconstruction: The mode of the distribution p_\theta(x | z) as a candidate reconstruction
                      Size: (batch_size, 3, 32, 32)
      Conditional Negative Log-Likelihood: The negative log-likelihood of the input x under the distribution p_\theta(x | z)
                                            Size: (batch_size,)
      KL: The KL Divergence between the variational approximate posterior with N(0, I)
          Size: (batch_size,)
    """
    posterior = self.encode(x)
    latent_z = posterior.sample()
    recon = self.decode(latent_z)

    return recon.mode(), recon.nll(x), posterior.kl()

class ToyEncoder(nn.Module):
  """
  Encoder for the toy dataset
  """
  def __init__(self, size, device):
    super(ToyEncoder, self).__init__()
    self.device = device

    self.encoder = nn.Sequential(
      nn.Linear(size, 64),
      nn.LeakyReLU(0.2, True),

      nn.Linear(64, 64),
      nn.LeakyReLU(0.2, True),

      nn.Linear(64, 64),
      nn.LeakyReLU(0.2, True),
    )

  def forward(self, inputs):
    batch_size = inputs.size(0)
    inputs = inputs.view(batch_size, -1)
    hidden = self.encoder(inputs)
    return hidden

class ToyDecoder(nn.Module):
  """
  Decoder for the toy dataset
  """
  def __init__(self, n_z):
    super(ToyDecoder, self).__init__()

    self.decoder_net = nn.Sequential(
      nn.Linear(n_z, 64),
      nn.LeakyReLU(0.2, True),

      nn.Linear(64, 64),
      nn.LeakyReLU(0.2, True),

      nn.Linear(64, 4),
      nn.LeakyReLU(0.2, True),
    )

  def forward(self, input):
    output = self.decoder_net(input)
    output = output.view(input.shape[0],2,2)
    return output

class Encoder(nn.Module):
  """
  Encoder for the MNIST dataset
  """
  def __init__(self, n_channels, n_filters):
    super(Encoder, self).__init__()

    # Encoder: (n_channels, size, size) -> (n_filters*8, size//16, size//16)
    self.encoder = nn.Sequential(
      nn.Conv2d(n_channels, n_filters, 4, 2, padding=1),
      nn.LeakyReLU(0.2, True),
      nn.BatchNorm2d(n_filters),

      nn.Conv2d(n_filters, n_filters * 2, 4, 2, padding=1),
      nn.LeakyReLU(0.2, True),
      nn.BatchNorm2d(n_filters * 2),

      nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, padding=1),
      nn.LeakyReLU(0.2, True),
      nn.BatchNorm2d(n_filters * 4),

      nn.Conv2d(n_filters * 4, n_filters * 8, 4, 2, padding=1),
      nn.LeakyReLU(0.2, True),
      nn.BatchNorm2d(n_filters * 8),
    )

  def forward(self, inputs):
    batch_size = inputs.size(0)
    hidden = self.encoder(inputs)
    hidden = hidden.view(batch_size, -1)
    return hidden

class Decoder(nn.Module):
  """
  Decoder for the MNIST dataset
  """
  def __init__(self, n_channels, n_filters, n_z, size):
    super(Decoder, self).__init__()
    self.n_filters = n_filters
    self.out_size = size // 4
    self.decoder_dense = nn.Sequential(
      nn.Linear(n_z, n_filters * 4 * self.out_size * self.out_size),
      nn.ReLU(True)
    )

    self.decoder_conv = nn.Sequential(
      nn.UpsamplingNearest2d(scale_factor=2),
      nn.Conv2d(n_filters * 4, n_filters * 2, 3, 1, padding=1),
      nn.LeakyReLU(0.2, True),

      nn.UpsamplingNearest2d(scale_factor=2),
      nn.Conv2d(n_filters * 2, n_filters, 3, 1, padding=1),
      nn.LeakyReLU(0.2, True),

      nn.UpsamplingNearest2d(scale_factor=2),
      nn.Conv2d(n_filters, n_channels, 3, 2, padding=1)
    )

  def forward(self, input):
    batch_size = input.size(0)
    hidden = self.decoder_dense(input).view(batch_size, self.n_filters * 4,
                                            self.out_size, self.out_size)
    output = self.decoder_conv(hidden)
    return output

class ToyVAE(VAE):
  """
  VAE for the toy dataset
  """
  def __init__(self, z_dim=100, input_size=32, device=torch.device("cuda:0")):
    super(ToyVAE, self).__init__(feature_size=64,
                                 encoder=ToyEncoder(size=input_size,
                                                    device=device),
                                 decoder=ToyDecoder(n_z=z_dim),
                                 z_dim=z_dim,
                                 device=device
                                 )

class MnistVAE(VAE):
  """
  VAE for the MNIST dataset
  """
  def __init__(self, in_channels=3, decoder_features=32, encoder_features=32,
               z_dim=100, input_size=32, device=torch.device("cuda:0")):
    out_size = input_size // 16
    super(MnistVAE, self).__init__(feature_size=encoder_features * 8 * out_size * out_size,
                                    encoder=Encoder(n_channels=in_channels,
                                                    n_filters=encoder_features),
                                    decoder=Decoder(n_channels=in_channels,
                                                    n_filters=decoder_features,
                                                    n_z=z_dim,
                                                    size=input_size),
                                    z_dim=z_dim,
                                    device=device
                                    )
