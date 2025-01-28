import math, tqdm, json
import torch
from  torch.nn.functional import normalize as torchNorm
from torch.nn import Module
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, iv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import SphericalVoronoi, geometric_slerp
import argparse

from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
from src.sampler.data import Data

c_sphere = "#8338EC"
c_points = "#8338EC"
c_edges = "#FB5607"
c_surface = "#FF006E"
l_size = 30
lw = 1.5
n_A = 21

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    N = 50
    d = 3
    alpha = 0.9
    dev = "cpu"
    if args.shuffle:
        data = Data(N, d, alpha, dev)
        points = data.X.numpy()
    else:
        points = np.load("resources/3D_points_exemple.npy")
    radius = 1
    center = np.array([0, 0, 0])
    sv = SphericalVoronoi(np.array(points), radius, center)

    T_tilde = 2*points[n_A] + sv.vertices[sv.regions[n_A][0]]
    T_tilde_norm = np.linalg.norm(T_tilde)
    T_tilde = T_tilde/T_tilde_norm
    T = np.array([0,0,1])
    O = np.array([0,0,0])
    A = points[n_A]
    sv.sort_vertices_of_regions()
    t_vals = np.linspace(0, 1, 2000)
    fig = plt.figure(figsize=(180,100))
    ax = fig.add_subplot( projection='3d')
    theta = ax.elev/360 * (2*np.pi)
    phi = ax.azim/360 * (2*np.pi)
    focal = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)] # coordinates of camera
    # plot the unit sphere for reference (optional)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=c_sphere, alpha=0.05)
    # plot generator points
    ax.scatter(0, 0, 1, c='k')
    ax.scatter(0, 0, 0, c='k')
    ax.scatter(A [0], A[1], A[2], c=c_points)
    ax.scatter(T_tilde[0], T_tilde[1], T_tilde[2], c='k')

    for j, region in enumerate(sv.regions):
        if j == n_A:
            alpha=1.0
            n = len(region)
            results = np.zeros(shape=(0,3))
            for i in range(n):
               start = sv.vertices[region][i]
               end = sv.vertices[region][(i + 1) % n]
               result = geometric_slerp(start, end, t_vals)
               ax.plot(result[..., 0],
                       result[..., 1],
                       result[..., 2],
                       c=c_edges, alpha=alpha)
               results = np.concatenate([results, result])
            polygon = Poly3DCollection([results],alpha=0.2)
            polygon.set_color(c_surface)
            ax.add_collection3d(polygon)
        else:
            n = len(region)
            alpha = 0.2
            lw_h= lw
            for i in range(n):
               start = sv.vertices[region][i]
               end = sv.vertices[region][(i + 1) % n]
               result = geometric_slerp(start, end, t_vals)
               if np.dot(focal, result.T).mean() >= 0:
                   alpha=0.8
                   lw_h=1.5 * lw
               ax.plot(result[..., 0],
                       result[..., 1],
                       result[..., 2],
                       c=c_edges, alpha=alpha, linewidth=lw_h)
            ax.scatter(points[j, 0], points[j, 1], points[j, 2], c=c_points, alpha=alpha)

    ax.plot([O[0], T[0]], [O[1], T[1]], [O[2], A[2]], linestyle="-", color="k")
    ax.plot([O[0], T[0]], [O[1], T[1]], [A[2], T[2]], linestyle="--", color="k")
    ax.plot([A[0], T[0]], [A[1], T[1]], [A[2], A[2]], linestyle="--", color="k")

    ax.set_aspect('equal')
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])
    _ = ax.set_zticks([])
    fig.set_size_inches(16, 16)
    ax.text(A[0]-0.0, A[1], A[2] + 0.05,'A', size=l_size)


    ax.text(T_tilde[0], T_tilde[1], T_tilde[2]+ 0.03,r'$\tilde{V}$', size=l_size)
    ax.text(-0.03, 0, 1.14, 'V', size=l_size)
    ax.text(0.03, 0, 0.05, 'O', size=l_size)
    ax.text(-0.11, 0, A[2]/2, r'$<V,A>$', size=l_size/3 * 2)
    ax.grid(False)
    ax.axis('off')

    plt.savefig("resources/3D_voronoi.png")
    plt.show()