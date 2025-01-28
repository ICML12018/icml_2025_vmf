import math, tqdm, json, os
import numpy as np
import argparse
import logging
from src.sampler.utils import set_gpus, read_glove_embeddings
from src.sampler.vmf_sampler import VMFSampler, SMAXSampler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
"""
This script performs Monte Carlo estimation of von Mises-Fisher (vMF) and Boltzmann probabilities
on a given dataset of Glove embeddings. The results are saved to disk for further analysis.
Usage:
    python compute_real_world.py -k <kappa> -a <alpha> -n <name> -bs <batch_size> -nt <n_trials>
Arguments:
    -k, --kappa (float): Concentration parameter for the vMF distribution.
    -a, --alpha (float): Similarity with the main embedding.
    -n, --name (str): Name of the dataset. Choices are ["glove-25"].
    -bs, --bs (int): Batch size for the experiments.
    -nt, --n_trials (int): Number of iterations for the experiments.
The script performs the following steps:
1. Parses command-line arguments.
2. Sets up the experiment path and GPU device.
3. Reads the Glove embeddings dataset.
4. Limits the dataset size for faster experiments.
5. Runs Monte Carlo estimation of vMF probabilities.
6. Runs Monte Carlo estimation of Boltzmann probabilities.
7. Saves the results and parameters to disk.
Functions:
    set_gpus(): Sets the GPU device for computation.
    read_glove_embeddings(d): Reads the glove embeddings dataset.
    VMFSampler: Class for sampling from the vMF distribution.
    SMAXSampler: Class for sampling from the Boltzmann distribution.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--kappa", type=float, required=True,
                        help="Concentration parameter")
    parser.add_argument("-a", "--alpha", type=float, required=True,
                        help="Similarity with main embedding")
    parser.add_argument("-n", "--name", type=str, required=True,
                        help="Name of dataset", choices=["glove-25"])
    parser.add_argument("-bs", "--bs", type=int, required=True,
                        help="Batch size")
    parser.add_argument("-nt", "--n_trials", type=int, required=True,
                        help="Number of iterations")

    args = parser.parse_args()
    exp_path = "results/%s/k=%.1f_a=%.2f_samples=%d/" % (args.name, args.kappa, args.alpha, args.bs*args.n_trials)
    os.makedirs(exp_path, exist_ok=True)

    dev = set_gpus()
 
    d = int(args.name.split("-")[-1])
    data = read_glove_embeddings(d)

    # limit data for faster experiments
    N=data.shape[0]
    perm = np.random.permutation(N)
    data = data[perm][:10**5]
    # Run Monte Carlo estimation of vMF probabilities
    vmf = VMFSampler(dev)
    exp = vmf.experiment_real_world(args.kappa, data, args.alpha, args.n_trials, args.bs)
    np.save(exp_path +"/data", exp)

    # Run Monte Carlo estimation of Boltzmann probabilities

    smSampler = SMAXSampler(dev)
    exp_sm = smSampler.experiment_real_world(args.kappa, data, args.alpha, min(args.n_trials, 10**3) , args.bs) # std is always narrow so speed up experiments
    np.save(exp_path + "/data_smax", exp_sm)

    N = data.shape[0]
    d = data.shape[1]
    params = {"k": args.kappa,
              "a": args.alpha,
              "d": d,
              "N": N,
              "name": args.name,
              "samples": args.bs * args.n_trials}
    with open(exp_path + 'params.json', 'w') as f:
        json.dump(params, f)