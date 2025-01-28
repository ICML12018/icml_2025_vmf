import math, tqdm, json, os
import numpy as np
import argparse
import logging
from src.sampler.utils import set_gpus
from src.sampler.vmf_sampler import VMFSampler, SMAXSampler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--kappa", type=float, required=True,
                        help="Concentration parameter")
    parser.add_argument("-a", "--alpha", type=float, required=True,
                        help="Similarity with main embedding")
    parser.add_argument("-d", "--d", type=int, required=True,
                        help="Dimension of embeddings")
    parser.add_argument("-N", "--N", type=int, required=True,
                        help="Number of point")
    parser.add_argument("-bs", "--bs", type=int, required=True,
                        help="Batch size")
    parser.add_argument("-nt", "--n_trials", type=int, required=True,
                        help="Batch size")

    args = parser.parse_args()
    exp_path = "results/uniform/k=%.1f_a=%.2f_d=%d_N=%d_samples=%d/" % (args.kappa, args.alpha, args.d, args.N, args.bs*args.n_trials)
    os.makedirs(exp_path, exist_ok=True)
    params = {"k": args.kappa,
              "a": args.alpha,
              "d": args.d,
              "N": args.N,
              "samples": args.bs * args.n_trials}
    with open(exp_path+'params.json', 'w') as f:
        json.dump(params, f)
    dev = set_gpus(0.20)

    # Run Monte Carlo simulation for vMF Exploration
    logging.info("Run Monte Carlo simulation for vMF Exploration on Uniform Data")
    vmf = VMFSampler(dev)
    exp = vmf.experiment(args.kappa, args.N, args.d, args.alpha, args.n_trials, args.bs)
    np.save(exp_path +"/data", exp)

    # Run simulation for Boltzmann Exploration
    logging.info("Run simulation for Boltzmann Exploration on Uniform Data")

    smSampler = SMAXSampler(dev)
    exp_sm = smSampler.experiment(args.kappa, args.N, args.d, args.alpha, args.n_trials)
    np.save(exp_path + "/data_smax", exp_sm)
    
    logging.info("Done! Probabilities saved in : %s" % exp_path)

    
    