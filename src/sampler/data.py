import torch
from src.sampler.utils import sample_uniformly
from src.sampler.utils import sample_specific_dot_product

class Data:
    def __init__(self, N, d, alpha, dev, batch_size=1):
      self.dev = dev
      # Create N vectors of size d uniformly distributed on the sphere
      self.X = sample_uniformly(N, d).to(dev).to(torch.float32)
      self.mu = sample_uniformly(1, d).to(dev)
      self.A = sample_specific_dot_product(self.mu, alpha, batch_size, dev).to(dev)
      self.mu = self.mu.squeeze().to(torch.float32)
      self.A = self.A.reshape((batch_size, d)).to(torch.float32)