import torch
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence

class DiagonalGaussian:
    """
    Gaussian Distribution with diagonal covariance matrix.
    """

    def __init__(self, mean, logvar = None, device = 'cpu'):
        """
        Initialize the DiagonalGaussian.

        Parameters:
            mean (torch.Tensor): The mean of the distribution.
            logvar (torch.Tensor, optional): Logarithm of the variance for each dimension. Defaults to zero tensor.
            device (str): The device ('cpu' or 'cuda') for computation. Defaults to 'cpu'.
        """
        self.device = device
        self.mean = mean.to(device)
        if logvar is None:
            logvar = torch.zeros_like(self.mean)
        self.logvar = torch.clamp(logvar, -30., 20.)

        self.std = torch.exp(self.logvar / 2)
        self.var = torch.exp(self.logvar)

        shape = mean.reshape(self.mean.shape[0], -1).shape
        self.dist = MultivariateNormal(mean.reshape(shape), torch.diag_embed(self.var.reshape(shape)))
        self.standard_normal = MultivariateNormal(torch.zeros(shape), torch.eye(shape[1]))

    def mode(self):
        """
        Return the mode of the distribution.

        Returns:
            torch.Tensor: The mode of the distribution, which is the mean.
        """
        return self.mean

    def sample(self):
        """
        Provide a sample from the distribution.

        Returns:
            torch.Tensor: Sampled tensor of the same size as the mean.
        """
        return self.mean + torch.randn(self.mean.shape).to(self.device)*self.std

    def nll(self, sample):
        """
        Compute the negative log likelihood of the sample under the distribution.

        Parameters:
            sample (torch.Tensor): A sample for which to compute the log-likelihood.

        Returns:
            torch.Tensor: Negative log-likelihood for each element in the batch.
        """
        return -self.dist.log_prob(sample.reshape(self.mean.shape[0], -1))

    def kl(self):
        """
        Compute the KL-Divergence between this distribution and the standard normal N(0, I).

        Returns:
            torch.Tensor: KL-Divergence for each element in the batch (size: batch size).
        """
        return kl_divergence(self.dist, self.standard_normal)
