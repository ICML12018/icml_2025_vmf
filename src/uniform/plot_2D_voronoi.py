import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
from src.sampler.data import Data

# set input data


def cart2pol(x, y):
    phi = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)
    return(phi, rho)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

h,w = 10,7
p_size = 10*w
t_size = 4.6*w
n_A = 8
loose_dash = (0, (5, 10))
c_sphere = "#8338EC"
c_points = "#8338EC"
c_edges = "#FB5607"
c_surface = "#FF006E"
dev = "cpu"
N = 11
d = 2
lw = 1.5

alpha = 0.9
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    if args.shuffle:
        data = Data(N, d, alpha, dev)
        points = data.X.numpy()
    else:
        points = np.load("resources/2D_points_exemple.npy")

    theta = np.linspace(0, 2 * math.pi, 5000)

    fig = plt.figure(figsize=(h, w))

    p_theta, p_r = cart2pol(points[:,0], points[:,1])
    p_theta = np.roll(np.sort(p_theta), -n_A)

    p_theta[1] += math.pi/12
    p_theta[2] += math.pi/12

    p_theta[0] += math.pi/12

    p_theta[-2] += math.pi/8
    p_theta[-3] += math.pi/10
    p_theta[-4] += math.pi/12
    edges = (p_theta + np.roll(p_theta, 1))/2
    edges[3] -= math.pi
    O = np.array([0,0])
    A = p_theta[0]

    # plot generator points

    ax = fig.add_subplot(projection="polar")
    ax.scatter(0, 1, s=p_size, c='k')
    ax.set_rmax(1.1)
    ax.set_rlim(0,1.1)
    ax.set_rticks([0])  # radial ticks
    ax.set_xticks([])  # angular ticks
    ax.spines['polar'].set_visible(False)

    theta_border = np.linspace(0, 2*math.pi, 5000)
    r_border = [1 for i in theta]
    ax.plot(theta_border, r_border, c=c_sphere, alpha=0.4, lw=lw)
    voronoi = np.linspace(edges[0], edges[1], 5000)
    ax.plot(voronoi, r_border, c_surface, alpha=0.6, lw=5)


    ax.scatter(p_theta, p_r, s=p_size/2, c = c_points, alpha=1.0, zorder=10)
    ax.scatter(edges, p_r, s=p_size, c = c_edges, marker="|",  alpha=1.0)
    ax.scatter(edges[0:3], p_r[0:3], s=3*p_size, c = c_edges, marker="|",  alpha=1.0)

    for i in range(1,N):
        ax.text(p_theta[-i], 1.04, r"$X_{%d}$" % i , size=3*t_size/4)

    ax.scatter(0, 0, s=p_size, c = 'k',  alpha=1.0, zorder=10)
    ax.text(math.pi/2 - 0.8, 0.03,'O', size=t_size)

    ax.scatter(math.pi/2,1, s=p_size, c = 'k', zorder=10)
    ax.text(math.pi/2 + 0.015, 1.04, 'V', size=t_size)

    ax.scatter(A, 1, s=p_size, c=c_points, alpha=1.0, zorder=10)
    ax.text(A, 1.05,'A', size=t_size, zorder=10)

    ax.scatter((math.pi/2 + 2*A)/3,1, s=p_size/2, c = 'k', zorder=10)
    ax.text((math.pi/2 + 2*A)/3, 1.04, r'$\tilde{V}$', size=3*t_size/4)

    ax.plot([0, math.pi/2], [0, 1], linestyle="-", color="k")
    ax.plot([0, A], [0, 1], linestyle="-", color="k")
    ax.plot([0, edges[0]], [0, 1], linestyle="-", color=c_edges, lw=0.5)
    ax.plot([0, edges[1]], [0, 1], linestyle="-", color=c_edges, lw=0.5)
    ax.plot([0, edges[0]], [0, 1], linestyle="-", color=c_surface, lw=0.5)
    ax.plot([0, edges[1]], [0, 1], linestyle="-", color=c_surface, lw=0.5)
    ax.plot([0, p_theta[-1]], [0, 1], linestyle=loose_dash, color="k")
    ax.plot([0, p_theta[1]], [0, 1], linestyle=loose_dash, color="k")


    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line

    # beta angles
    n_lin = 5000
    ax.plot(np.linspace(A, p_theta[-1], n_lin), [0.4 for i in range(n_lin)], linestyle="--", c='k', alpha=1.0, lw=1)
    ax.plot(np.linspace(math.pi/2, A, n_lin), [0.6 for i in range(n_lin)], linestyle="--", c='k', alpha=1.0, lw=1)
    ax.plot(np.linspace(math.pi/2, edges[0], n_lin), [0.8 for i in range(n_lin)], linestyle="--", c='k', alpha=1.0, lw=1)

    ax.text(A, 1.05,'A', size=t_size, zorder=10)
    ax.scatter(A , 0.6, marker=">", c='k')
    ax.scatter(p_theta[-1] , 0.4, marker=">", c='k')
    ax.scatter(edges[0] , 0.8, marker="<", c='k')

    ax.text((A +math.pi/2)/2, 0.65, r"$\theta_0$", size=2*t_size/3)
    ax.text((A +p_theta[-1])/2, 0.45, r"$\beta_1$", size=2*t_size/3)
    ax.text((4 * edges[0] +math.pi/2)/5, 0.85, r"$\theta_0 + \frac{\beta_1}{2}$", size=2*t_size/3)

    ax.grid(False)
    plt.savefig("resources/2D_voronoi.png")
    plt.show()