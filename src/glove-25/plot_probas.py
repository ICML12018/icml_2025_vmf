import matplotlib.pyplot as plt
"""
This script generates a plot based on real-world data and theoretical probabilities.

The script performs the following steps:
1. Parses command-line arguments to get the input data path.
2. Loads parameters from a JSON file and data from NumPy files.
3. Computes theoretical probabilities using the `theo_proba` function.
4. Configures the plot appearance using Seaborn and Matplotlib.
5. Plots the theoretical and empirical probabilities.
6. Saves the plot as a PNG file.

Command-line arguments:
    -p, --path: str, required
        Path to the directory containing the input data files.

Input files:
    params.json: JSON file containing parameters 'k', 'a', 'd', 'N', 'samples', and 'name'.
    data.npy: NumPy file containing empirical data.
    data_smax.npy: NumPy file containing empirical data for softmax.

Output:
    plot.png: PNG file containing the generated plot.

Raises:
    ValueError: If the dataset name is unknown.
"""
import numpy as np
import argparse
import json
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from src.sampler.utils import theo_proba


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True,
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
    name = params["name"]
    n_range = np.arange(1, int(N) + 1)

    sns.set()
    p = data / ns
    #ci = 1.96 * np.sqrt(p * (1 - p) / (ns * n_range))

    fig, ax = plt.subplots()

    # Force scientific notation on the y-axis
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))  # Limits for switching to scientific notation
    ax.yaxis.set_major_formatter(formatter)

    SMALL_SIZE = 16
    MEDIUM_SIZE = 24
    BIGGER_SIZE = 24

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    zero_order = theo_proba(alpha, d, kappa, n_range, 0)
    first_order = theo_proba(alpha, d, kappa, n_range, 1)

    if name.startswith("glove"):
        s=10**5
        title= "Glove-%d" % d
    else:
        raise  ValueError("Unknown dataset")

    idx = np.arange(10**4, 10**5, 10**3)

    sns.lineplot(x=n_range[idx], y= zero_order[idx], ax=ax, color="orange", label=r"$P_{0}(a)$")
    sns.lineplot(x=n_range[idx], y= first_order[idx], ax=ax, color="green", label=r"$P_{1}(a)$")
    sns.lineplot(x=n_range[idx], y= data_smax[idx], ax=ax, color="red", label=r"$P_{B-exp}(a)$")
    sns.lineplot(x=n_range[idx], y= p[idx], ax=ax, color="blue", label=r"$P_{vMF-exp}(a)$")
    #sns.lineplot(x=n_range[idx], y= p[idx] /data_smax[idx], ax=ax, color="blue", label=r"$P_{vMF-exp}(a)$")


    #ax.fill_between(n_range[idx], (p-ci)[idx], (p+ci)[idx], color='blue', alpha=.1)
    plt.xlabel("n")
    plt.ylabel("P(a)")
    plt.legend()
    #plt.title(r"%s, $\kappa=%.1f$, <V,A>=%.1f," % (name, kappa, alpha))
    plt.title(("%s \n " % title) + "$\kappa=%.1f$, <V,A>=%.1f:" % (kappa, alpha))
    title_text = "%s \n " % title
    kappa_text = "$\kappa=%.1f$" % kappa
    alpha_text = "<V,A>=%.1f:" % alpha
    plt.title(title_text + kappa_text + ", " + alpha_text)
    plt.savefig(input_path + "/plot.png")