import math, tqdm, json
import torch
from torch.nn.functional import normalize as torchNorm
import numpy as np
import pandas as pd
from scipy.special import gamma, iv
from scipy import special
import GPUtil

def get_device():
    try:
        if torch.cuda.is_available():
            dev = torch.device('cuda')
        else:
            dev = torch.device('cpu')
    except:
        dev = 'cpu'
    return dev

def set_gpus(max_memory=0.05):
    """
        find gpu card available
    """
    print("TORCH version: {}".format(torch.__version__))
    gpu_index = GPUtil.getAvailable(limit=4, maxMemory=max_memory)
    # setting gpu for tensorflow
    try:
        gpu_index = [gpu_index[0]]
    except Exception as e:
        raise ValueError("No GPU available!!")
    print("\t Using GPUs: {}".format(gpu_index))

    device = "cuda:{}".format(",".join([str(i) for i in gpu_index]))
    return device


def sample_uniformly(n_vectors, dim):
    """
  Uniform sampling on S^{d-1} sphere
  """
    return torchNorm(torch.Tensor(np.random.normal(loc=0, scale=1, size=(n_vectors, dim))), dim=-1)


def sample_specific_dot_product(mu, alpha, n, dev):
    """
  Given an initial vector mu, sample n vectors whose similarity with mu is alpha
  """
    d = mu.shape[-1]
    bs = mu.shape[0]
    z = sample_uniformly(bs * n, d).to(dev)
    mu_repeat = torch.repeat_interleave(mu, n, dim=0)
    dp = (z * mu_repeat).sum(dim=-1)
    v = dp.unsqueeze(-1).repeat(1, d) * (mu_repeat)  # colinear
    z = z - v  # orthogonal

    y = alpha * mu_repeat + math.sqrt(1 - alpha ** 2) * z
    y = torchNorm(y, dim=-1)  # ||y|| isn't exactly 1 because of numerical stability
    return y.reshape(bs, n, d)


#
def smax_proba(mu, X, alpha, k):
    """
  Compute empirical softmax proba
  """
    N = X.shape[0]
    dot_products = torch.reshape(torch.matmul(X, mu.T), shape=(bs, int(N)))
    denom = 1 + torch.cumsum(torch.exp(k * (dot_products - alpha)), dim=1)
    res = (1 / denom).cpu().numpy()
    return res


def theo_proba(alpha, d, k, n_range, order=0):
    """
  Compute theoretical proba.
    - Order 0 gives the asymptotical proba for softmax, which is also the
  first term of the expansion of the VMF proba.
    - Order 1 adds the next term of the VMF expansion
  """
    mun = 1 - ((special.beta(1 / 2, (d - 1) / 2) * (d - 1)) ** (2 / (d - 1)) / 2) * special.gamma((d + 1) / (d - 1)) / (
                n_range ** (2 / (d - 1)))
    s = 1
    if order >= 1:
        term = k * alpha * (mun - 1)
        s += term

    coeff = (k / 2) ** (d / 2 - 1) / (gamma(d / 2) * iv(d / 2 - 1, k))
    theory_smax = math.exp(k * alpha) * coeff / (n_range) * s
    return theory_smax

def read_glove_embeddings(dim=25):
    """
    Reads GloVe embeddings from a specified file, normalizes them, and returns the embeddings on a unit sphere.

    Parameters:
    dim (int): The dimensionality of the GloVe embeddings to read. Default is 25.

    Returns:
    np.ndarray: A 2D numpy array where each row is a normalized GloVe embedding vector on a unit sphere.
    """
    df = pd.read_csv("dataset/glove.twitter.27B.%dd.txt" % dim, delimiter=" ", header=None)
    data = df.to_numpy()[:,1:].astype(np.float32)
    data_mean = data.mean(axis=0)[None,:]
    data_centered = data - data_mean
    data_norm = np.sqrt(np.sum(data_centered**2, axis=1))[:,None]
    data_sphere = data_centered/data_norm
    return data_sphere

