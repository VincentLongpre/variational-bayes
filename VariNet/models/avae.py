import torch
from torch import nn
import numpy as np
from ..utils.dists import DiagonalGaussian
from ..models.vae import ToyDecoder, Decoder

class AVAE(nn.Module):
  def __init__(self, adversary, encoder, decoder, z_dim=100, device=torch.device("cuda:0")):
    super(AVAE, self).__init__()

    self.z_dim = z_dim
    self.device = device
    self.adversary = adversary
    self.encoder = encoder
    self.decoder = decoder
    
  def encode(self, x):
    """
    Returns samples from the posterior distribution q_\phi(z | x) along with the mean and variance
    """
    return self.encoder(x)

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
    z = torch.randn((batch_size,self.z_dim))
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
    prior = DiagonalGaussian(torch.zeros_like(posterior.mean), device=self.device)

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
      Conditional Negative Log-Likelihood: The negative log-likelihood of the input x under the distribution p_\theta(x | z)
      KL estimate: The estimated KL Divergence between the variational approximate posterior with N(0, I) given by the adversary network
      Adversary loss: The loss function for the discriminator network
    """
    latent_z = self.encode(x)

    prior = DiagonalGaussian(torch.zeros(x.shape[0],latent_z.shape[1]).to(self.device), device=self.device)
    sampled_z = prior.sample()

    recon = self.decode(latent_z)

    T_d = self.adversary(x, latent_z)
    T_i = self.adversary(x, sampled_z)

    CE_loss = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    adv_loss = CE_loss(sigmoid(T_d), torch.ones_like(T_d)) + CE_loss(sigmoid(T_i), torch.zeros_like(T_i))

    return recon.mode(), recon.nll(x), T_d, adv_loss

class ToyEncoder(nn.Module):
  """
  Encoder network for the toy dataset - AVAE (noise here!)
  """
  def __init__(self, size, n_z, device):
    super(ToyEncoder, self).__init__()

    self.device = device
    self.encoder = nn.Sequential(
      nn.Linear(size + 64, 64),
      nn.LeakyReLU(0.2, True),

      nn.Linear(64, 64),
      nn.LeakyReLU(0.2, True),

      nn.Linear(64, 64),
      nn.LeakyReLU(0.2, True),

      nn.Linear(64, 64),
      nn.LeakyReLU(0.2, True),

      nn.Linear(64, n_z)
    )

  def forward(self, inputs):
    batch_size = inputs.size(0)
    eps = torch.randn((batch_size, 64)).to(self.device)

    inputs = torch.concat([inputs.view(batch_size, -1),eps], dim=1)
    output = self.encoder(inputs)

    return output

class Encoder(nn.Module):
  """
  Encoder network for MnistAVAE
  """
  def __init__(self, nc, nef, nz, isize, device, eps_basis = 16, eps_dim = 32):
    super(Encoder, self).__init__()

    self.eps_basis = eps_basis
    self.eps_dim = eps_dim

    # Device
    self.device = device

    # Encoder: (nc, isize, isize) -> (nef*8, isize//16, isize//16)
    self.encoder = nn.Sequential(
      nn.Conv2d(nc, nef, 4, 2, padding=1),
      nn.LeakyReLU(0.2, True),
      nn.BatchNorm2d(nef),

      nn.Conv2d(nef, nef * 2, 4, 2, padding=1),
      nn.LeakyReLU(0.2, True),
      nn.BatchNorm2d(nef * 2),

      nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1),
      nn.LeakyReLU(0.2, True),
      nn.BatchNorm2d(nef * 4),

      nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1),
      nn.LeakyReLU(0.2, True),
      nn.BatchNorm2d(nef * 8)
    )

    self.z0_projector = nn.Sequential(
      nn.Linear(nef * 8, nz),
      nn.LeakyReLU(0.2, True)
    )
  
    self.a_projector = nn.Sequential(
      nn.Linear(nef * 8, nz),
      nn.LeakyReLU(0.2, True)
    )

    self.v_network = nn.Sequential(
      nn.Linear(eps_dim, 128),
      nn.LeakyReLU(0.2, True),

      nn.Linear(128, 128),
      nn.LeakyReLU(0.2, True),

      nn.Linear(128, nz),
      nn.LeakyReLU(0.2, True)
    )

  def forward(self, inputs):
    batch_size = inputs.size(0)
    eps = torch.randn((self.eps_basis, batch_size, self.eps_dim)).to(self.device)

    net = self.encoder(inputs)
    net = net.view(batch_size,-1)

    a = self.a_projector(net)
    z0 = self.z0_projector(net)

    v_all = self.v_network(eps)  

    v_sum = torch.sum(v_all, dim=0)

    # Monte Carlo estimates for mean and variance
    z = z0 + v_sum
    Ez = z0 + a * torch.mean(v_all, dim=0)
    Varz = a * a * torch.var(v_all, dim=0)

    return z, Ez, Varz

  def forward(self, inputs):
    batch_size = inputs.size(0)
    eps = torch.randn((batch_size, 64)).to(self.device)

    inputs = torch.concat([inputs.view(batch_size, -1), eps], dim=1)
    output = self.encoder(inputs)

    return output

class ToyAdversary(nn.Module):
  """
  Adversary network of the ToyAVAE
  """
  def __init__(self, size, n_z, device):
    super(ToyAdversary, self).__init__()
    self.device = device

    self.input_net = nn.Sequential(
      nn.Linear(size+n_z, 256),
      nn.LeakyReLU(0.2, True)
    )

    self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 256),
                nn.LeakyReLU(0.2, True),
                nn.Linear(256, 256)
            ) for _ in range(5)
        ])

    self.out_net = nn.Linear(256, 1)

    self.activation = nn.LeakyReLU(0.2, True)

  def forward(self, x, z):
    inputs = torch.concat([x.view(x.shape[0],-1),z], dim=1)

    output = self.input_net(inputs)

    for block in self.blocks:
      scut = output
      output = block(output)
      output += scut
      output = self.activation(output)

    output = self.out_net(output)
    output = torch.squeeze(output)
    return output
  
class Adversary(nn.Module):
  """
  Adversary network of the MnistAVAE
  """
  def __init__(self,isize, nz):
    super(Adversary, self).__init__()

    self.isize = isize
    self.nz = nz

    self.theta_net = nn.Sequential(
      nn.Linear(isize * isize, 1024),
      nn.LeakyReLU(0.2, True),

      nn.Linear(1024, 1024),
      nn.LeakyReLU(0.2, True),

      nn.Linear(1024, 1024),
      nn.LeakyReLU(0.2, True),

      nn.Linear(1024, 8192)
    )

    self.s_net = nn.Sequential(
      nn.Linear(nz, 1024),
      nn.LeakyReLU(0.2, True),

      nn.Linear(1024, 1024),
      nn.LeakyReLU(0.2, True),

      nn.Linear(1024, 1024),
      nn.LeakyReLU(0.2, True),

      nn.Linear(1024, 8192)
    )

    self.z_stat = nn.Sequential(
      nn.Linear(nz, 1024),
      nn.LeakyReLU(0.2, True),

      nn.Linear(1024, 1024),
      nn.LeakyReLU(0.2, True),

      nn.Linear(1024, 1024),
      nn.LeakyReLU(0.2, True),

      nn.Linear(1024, 1)
    )

    self.x_stat = nn.Sequential(
      nn.Linear(isize * isize, 1024),
      nn.LeakyReLU(0.2, True),

      nn.Linear(1024, 1024),
      nn.LeakyReLU(0.2, True),

      nn.Linear(1024, 1024),
      nn.LeakyReLU(0.2, True),

      nn.Linear(1024, 1)
    )

  def forward(self, x, z):
    x = x.view(x.shape[0],-1)
    theta = self.theta_net(x)
    s = self.s_net(z)
    T_x = self.x_stat(x)
    T_z = self.z_stat(z)

    output = torch.sum(theta * s, dim=1, keepdim=True) + T_x + T_z
    return output
  
class ToyAVAE(AVAE):
  """
  AVAE for the toy dataset
  """
  def __init__(self, z_dim=100, input_size=32, device=torch.device("cuda:0")):
    super(ToyAVAE, self).__init__(adversary=ToyAdversary(size=input_size,
                                                         n_z=z_dim,
                                                         device=device),
                                  encoder=ToyEncoder(size=input_size,
                                                     device=device,
                                                     n_z=z_dim),
                                  decoder=ToyDecoder(n_z=z_dim),
                                  z_dim=z_dim,
                                  device=device
                                 )

class MnistAVAE(AVAE):
  """
  AVAE for the MNIST dataset
  """
  def __init__(self, in_channels=3, decoder_features=32, encoder_features=32,
               z_dim=100, input_size=32, device=torch.device("cuda:0")):
    super(MnistAVAE, self).__init__(feature_size=64,
                                    adversary=Adversary(size=input_size,
                                                        nz=z_dim,
                                                        device=device),
                                    encoder=Encoder(size=input_size,
                                                    device=device),
                                    decoder=Decoder(nc=in_channels, 
                                                    ndf=decoder_features, 
                                                    nz=z_dim, 
                                                    isize=input_size
                                                    )
                                      )