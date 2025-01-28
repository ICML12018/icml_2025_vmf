import matplotlib.pyplot as plt

import numpy as np
from src.utils import theo_proba
import argparse
import json
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

"""
This script compares the Boltzmann and von Mises-Fisher (vMF) distributions by plotting their probabilities.

Usage:
    compare_boltzmann_vs_vmf.py --values <comma-separated values> --k <float>

Arguments:
    --values: Comma-separated list of values for which probabilities are computed. Default is '0.90,0.30,0.0,-0.30,-0.90'.
    --k: Value of k (float). Default is 1.0.

Description:
    The script loads precomputed probability data for the specified values and k from .npy files.
    It then generates two plots:
    1. vMF-exponential probabilities as a function of sample size.
    2. Boltzmann-exponential probabilities as a function of sample size.

    The plots are saved as 'results/vmf_plot_<k>.pdf' and 'results/boltzmann_plot.pdf' respectively.

Dependencies:
    - matplotlib
    - numpy
    - seaborn
    - argparse
    - json
    - src.utils (for theo_proba function)

Example:
    python compare_boltzmann_vs_vmf.py --values '0.90,0.30,0.0,-0.30,-0.90' --k 1.0
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare Boltzmann vs vMF.')
    parser.add_argument('--values', type=str, default='0.90,0.30,0.0,-0.30,-0.90', help='Comma-separated list of values. Must compute the probabilities for each value first using compute_probas.py')
    parser.add_argument('--k', type=float, default=1.0, help='Value of k')
    args = parser.parse_args()

    values = [float(v) for v in args.values.split(',')]
    k = args.k

    data = [np.load("results/k=%.1f_a=%.2f_samples=30000000/data.npy" % (k, v)) for v in values]
    datas_smax = [ np.load("results/k=%.1f_a=%.2f_samples=30000000/data_smax.npy" % (k, v)) for v in values]

    ns = 3 * 10**7 # number of samples



    sns.set()
    # Force scientific notation on the y-axis
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))  # Limits for switching to scientific notation


    SMALL_SIZE = 16
    MEDIUM_SIZE = 24
    BIGGER_SIZE = 24



    idx = np.arange(2 * 10** 4, 10 ** 5, 10 ** 3) #range of x-axis
    n_range = 1 + np.array(range(int(data[0].shape[0])))
    cmap = sns.light_palette("blue", n_colors=len(values) +3, as_cmap=False)

    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(formatter)
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    for i,v in enumerate(values):
        x = n_range[idx]
        y = data[i][idx]/ns
        sns.lineplot(x=x, y=y, ax=ax, color=cmap[-(i+1)])
        plt.text(x[0] + 0.2, y[0], "%.1f" % v, fontsize=10, va='center')
    plt.xlabel("n")
    plt.ylabel(r"$P_{vMF-exp}(a)$")
    #plt.title(r"Glove-25, $\kappa$=%d :" % int(k)+ "\n $P_{vMF-exp}(a)$ increases with <V,A> ")
    plt.suptitle("Glove-25, vMF-exp", fontsize=SMALL_SIZE)
    plt.title(r"$\kappa$=%d , <V,A> $\in$ [-0.9,0.9]" % int(k), fontsize=MEDIUM_SIZE/2)
    plt.ylim(0)
    plt.savefig("results/vmf_plot.png")





    sns.set()

    cmap = sns.light_palette("red", n_colors=len(values)+3, as_cmap=False)
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(formatter)
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    for i,v in enumerate(values):
        x = n_range[idx]
        y = datas_smax[i][idx]
        sns.lineplot(x=x, y=y, ax=ax, color=cmap[-(i+1)])
        plt.text(x[0] + 0.2, y[0], "%.1f" % v, fontsize=10, va='center')

    plt.xlabel("n")
    plt.ylabel(r"$P_{B-exp}(a)$")
    plt.suptitle("Glove-25, B-exp", fontsize=SMALL_SIZE)
    plt.title(r"$\kappa$=%d , <V,A> $\in$ [-0.9,0.9]" % int(k), fontsize=MEDIUM_SIZE / 2)
    plt.ylim(0)
    plt.savefig("results/boltzmann_plot.png")