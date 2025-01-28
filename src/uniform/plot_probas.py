import matplotlib.pyplot as plt
import numpy as np
from src.sampler.utils import theo_proba
import argparse
import json
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", "--path", type=str, required=True,
                        help="path for data")

    args = parser.parse_args()
    input_path = args.path
    with open(input_path + "/params.json") as f:
        params = json.load(f)

    data = np.load(input_path + "/data.npy")
    data_smax = np.load(input_path + "/data_smax.npy")
    kappa = params["k"]
    alpha = params["a"]
    d = params["d"]
    N = params["N"]
    ns = params["samples"]

    sns.set()
    p = data / ns
    ci = 1.96 * np.sqrt(p * (1 - p) / ns)
    fig, ax = plt.subplots()
    s = 10**5

    SMALL_SIZE = 16
    MEDIUM_SIZE = 24
    BIGGER_SIZE = 24

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    n_range = 1 + np.array(range(int(N)))
    zero_order = theo_proba(alpha, d, kappa, n_range, 0)
    first_order = theo_proba(alpha, d, kappa, n_range, 1)
    n_range = 1 + np.array(range(int(N)))
    idx = np.arange(s, N, 10**3)
    sns.lineplot(x=n_range[idx], y= zero_order[idx], ax=ax, color="orange", label=r"$P_{0}(a)$")
    sns.lineplot(x=n_range[idx], y= first_order[idx], ax=ax, color="green", label=r"$P_{1}(a)$")
    sns.lineplot(x=n_range[idx], y= data_smax[idx], ax=ax, color="red", label=r"$P_{B-exp}(a)$")
    sns.lineplot(x=n_range[idx], y= p[idx], ax=ax, color="blue", label=r"$P_{vMF-exp}(a)$")

    ax.fill_between(n_range[idx], (p-ci)[idx], (p+ci)[idx], color='blue', alpha=.1)
    plt.legend()
    plt.xlabel("n")
    plt.ylabel("P(a)")
    plt.legend()
    plt.title(r"$\kappa=%.1f$, <V,A>=%.1f, d=%d " % (kappa, alpha, d))
    plt.ylim(0)
    plt.savefig(input_path + "/plot.png")