import numpy as np
from scipy.stats import invwishart
import matplotlib.pyplot as plt

#Initialization
mu0 = [0, 0]
cov0 = [[5, 0], [0, 5]]
dim = 2
df = 10
scale = np.eye(dim)
scale[0,1] = 0.5
scale[1,0] = 0.5

#K = Number of Gaussian, N = Number of data per Gaussian
K = 3
N = 1000
means = []
covariances = []

#Randomly generate K Gaussians
#means ~ Normal(mu0, cov0)
#covariances ~ Inverse Wishart(df, scale)
for i in range(K):
    mu = np.random.multivariate_normal(mu0, cov0)
    means.append(mu)
    #means = [[3.3, 3], [-2, 3.5], [1, -2.5]]
    iw = invwishart(df, scale)
    iw_rvs = invwishart.rvs(df, scale)*10
    covariances.append(iw_rvs)

print(means)
print(covariances)

#alpha ~ hyperparamenter of theta
#theta ~ Dirichlet(alpha)
alpha = [100] * K
theta = np.random.dirichlet(alpha=alpha, size=1)[0]

#Display
plt.ion()
fig, ax = plt.subplots()
x, y = [],[]
plt.xlim(-6,6)
plt.ylim(-6,6)
plt.draw()
colors = ['b', 'y', 'r', 'm', 'g', 'c']
makers = ['.', '*', 'o', 'v', '^', '8']
for i in range(K):
    x.append(means[i][0])
    y.append(means[i][1])
    plt.scatter(means[i][0], means[i][1], marker=makers[i], color=colors[i+3])
    fig.canvas.draw_idle()
    plt.pause(0.1)

#z ~ Categorical(theta)
for i in range(N):
    z = np.random.multinomial(n=1, pvals=theta)
    x_component = [j for j in range(K) if (z[j]==1)][0]
    x_data = np.random.multivariate_normal(means[x_component], covariances[x_component])
    #print(x_data)
    x.append(x_data[0])
    y.append(x_data[1])
    plt.scatter(x_data[0], x_data[1], marker=makers[x_component], color=colors[x_component])
    fig.canvas.draw_idle()
    plt.pause(0.05)

print(theta)
plt.waitforbuttonpress()