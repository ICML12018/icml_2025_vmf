import math, tqdm, json
import torch
import numpy as np
from src.sampler.data import Data
from src.sampler.utils import sample_specific_dot_product

class VMFSampler:
  def __init__(self, dev):
    self.dev = dev
    return

  def random_VMF (self, mu , kappa , size = None ):
    """
    Von Mises - Fisher distribution sampler with
    mean direction mu and concentration kappa .
    Source : https://hal.science/hal-04004568
    """
    # parse input parameters
    n = 1 if size is None else np . prod ( size )
    shape = () if size is None else tuple ( np . ravel ( size ))
    mu = np . asarray ( mu )
    mu = mu / np . linalg . norm ( mu )
    (d ,) = mu . shape
    # z component : radial samples perpendicular to mu
    z = np . random . normal (0 , 1 , (n , d) )
    z /= np . linalg . norm (z , axis =1 , keepdims = True )
    z = z - (z @ mu[:, None ]) * mu[None , :]
    z /= np . linalg . norm (z , axis =1 , keepdims = True )
    # sample angles ( in cos and sin form )
    cos = self.random_VMF_cos (d , kappa , n )
    sin = np . sqrt (1 - cos ** 2)
    # combine angles with the z component
    x = z * sin [:, None ] + cos [:, None ] * mu[None , :]
    return x. reshape ((*shape , d ))

  def random_VMF_cos (self, d: int , kappa : float , n: int):
    """
    Generate n iid samples t with density function given by
    p(t) = someConstant * (1-t**2) **((d-2)/2) * exp ( kappa *t)
    Source : https://hal.science/hal-04004568

    """
    # b = Eq. 4 of https :// doi . org / 10. 1080 / 0 3 6 1 0 9 1 9 4 0 8 8 1 3 1 6 1
    b = ( d - 1) / (2 * kappa + (4 * kappa ** 2 + ( d - 1) ** 2) ** 0.5)
    x0 = (1 - b) / ( 1 + b)
    c = kappa * x0 + ( d - 1) * np . log (1 - x0 ** 2)
    found = 0
    out = []
    while found < n:
      m = min(n , int(( n - found ) * 1.5))
      z = np . random . beta (( d - 1 ) / 2 , (d - 1) / 2 , size =m )
      t = ( 1 - (1 + b) * z) / (1 - (1 - b) * z)
      test = kappa * t + (d - 1) * np . log (1 - x0 * t) - c
      accept = test >= -np . random . exponential ( size =m)
      out . append (t[ accept ])
      found += len ( out [-1])
    return np . concatenate ( out )[:n]

  def experiment(self, kappa, N, d, alpha, n_trials, bs):
    probas_trials = np.zeros(N)
    for e in tqdm.tqdm(range(n_trials)):
        data = Data(N, d, alpha, self.dev)
        mu_sampled = self.random_VMF(data.mu.cpu(), kappa, size=bs)
        mu_sampled = torch.Tensor(mu_sampled).to(self.dev).to(torch.float16)
        # count the average number of times when A is the nearest neighbour
        candidates = torch.cat([data.A, data.X], dim=0).to(torch.float16)
        dps = torch.matmul(mu_sampled, candidates.T).squeeze()
        diffs = (dps[:, 1:] > dps[:, 0][:,None])
        egals = (dps[:, 1:] == dps[:, 0][:,None])
        probas = ((torch.cumsum(diffs, dim=-1) == 0)*(1/(1+(torch.cumsum(egals, dim=-1))))).sum(dim=0).float().squeeze().cpu().numpy()
        probas_trials += probas
    return probas_trials

  def experiment_alt(self, kappa, N, d, alpha, n_trials, bs):
    probas_trials = np.zeros(N)
    for e in tqdm.tqdm(range(n_trials)):
        data = Data(N, d, alpha, self.dev)
        mu_sampled = self.random_VMF(data.mu.cpu(), kappa, size=bs)
        mu_sampled = torch.Tensor(mu_sampled).to(self.dev).to(torch.float16)
        # count the average number of times when A is the nearest neighbour
        candidates = torch.cat([data.A, data.X], dim=0).to(torch.float16)
        dps = torch.matmul(mu_sampled, candidates.T).squeeze()
        all_k = (dps[:,1:] > dps[:,0][:,None]).sum(dim=1) +1
        max_k = all_k.max()
        probas = torch.zeros((bs, N)).to(self.dev)
        probas[:,0] = (N - all_k) / N
        for i in range(1, N - max_k):
          probas[:,i] = probas[:,i - 1] * (N - all_k - i) / (N - i)

        probas = torch.clamp(probas, min=0).sum(dim=0).cpu().numpy()
        probas_trials += probas

    return probas_trials # returns the number of times A is the nearest neighbour. Must divide by 1 + range(1, N + 1) to get the probability.

  def experiment_real_world(self, kappa, data, alpha, n_trial, bs):
      data = torch.Tensor(data).to(self.dev)
      N = data.shape[0]
      d = data.shape[1]
      probas_trials = np.zeros(N-2)

      for e in tqdm.tqdm(range(n_trial)):
          perm = torch.randperm(N)
          X = data[perm]
          mu = torch.Tensor(X[0]).unsqueeze(0).to(torch.float32) # randomly select V in the embedding set
          candidates = X[1:]
          scores = torch.matmul(mu, candidates.T)
          idx_A = torch.argmin(torch.abs(scores-alpha)) # select A so that <A,V> == alpha
          candidates[[0, idx_A], :] = candidates[[idx_A, 0], :] # put A in first position of candidates list
          mu_sampled = self.random_VMF(mu.squeeze().cpu(), kappa, size=bs)
          mu_sampled = torch.Tensor(mu_sampled).to(self.dev).to(torch.float32)
          # count the average number of times when A is the nearest neighbour
          dps = torch.matmul(mu_sampled, candidates.T).squeeze()
          diffs = (dps[:, 1:] > dps[:, 0][:, None])
          probas = (torch.cumsum(diffs, dim=-1) == 0).sum(
              dim=0).float().squeeze().cpu().numpy()
          probas_trials += probas
      return probas_trials # returns the number of times A is the nearest neighbour. Must divide by 1 + range(1, N + 1) to get the probability.

class SMAXSampler:
  def __init__(self, dev):
    self.dev = dev
    return
  def experiment(self, kappa, N, d, alpha, n_trials):
    probas_trials = np.zeros(N)
    for e in tqdm.tqdm(range(n_trials)):
      data = Data(N, d, alpha, self.dev)
      probas_trials += self.smax_proba(data.mu, data.X, alpha, kappa)
    return probas_trials/n_trials

  def experiment_real_world(self, kappa, data, alpha, n_trial, bs):
    data = torch.Tensor(data).to(self.dev)
    N = data.shape[0]
    d = data.shape[1]
    probas_trials = np.zeros(N)
    for e in tqdm.tqdm(range(n_trial)):
        perm = torch.randperm(N)
        X = data[perm]
        mu = torch.Tensor(X[0])
        probas_trials += self.smax_proba(mu, data, alpha, kappa)
    return probas_trials / n_trial

  def smax_proba(self, mu, X, alpha, k):
    """
  Compute empirical softmax proba
  """
    dot_products = torch.matmul(X, mu.T)
    denom = 1 + torch.cumsum(torch.exp(k * (dot_products - alpha)), dim=0)
    res = (1 / denom).cpu().numpy()
    return res
