import matplotlib.pyplot as plt
import pylab as pyl
import numpy as np
from matplotlib.patches import Ellipse

"""
K-Means
"""


def data_display(X, K, label, title):
    fig = plt.figure()
    fig.suptitle(title + " K=" + str(K), fontsize=14, fontweight='bold')
    for k in range(K):
        data = X[label == k]
        plt.plot(data[:, 0], data[:, 1], '.')
    plt.axis('equal')
    plt.plot()
    plt.show()


"""
Gaussian Mixture Model
"""


def sorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def plot_ellipse(pos, cov, nstd=2, ax=None, **kwargs):
    if ax is None:
        ax = pyl.gca()
    vals, vecs = sorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(abs(vals))
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellip)
    return ellip


def GMM_data_display(X, label, mu, sigma, K, title):
    fig = plt.figure()
    fig.suptitle(title + " K=" + str(K), fontsize=14, fontweight='bold')
    for k in range(K):
        data = X[label == k]
        plt.plot(data[:, 0], data[:, 1], '.')
    colors = ['b', 'g', 'c', 'm', 'y', 'r']
    for k in range(K):
        plot_ellipse(mu[k], sigma[k], alpha=0.6, color=colors[k % len(colors)])
    plt.plot()
    plt.show()
