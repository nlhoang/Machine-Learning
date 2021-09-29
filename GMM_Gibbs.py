# ------------------------------------------------------------
# Gaussian Mixture Model - Gibbs Sampling
# ------------------------------------------------------------


import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import invwishart
from Function import DataFiles as df, GraphDisplay as gp


class GMM_Gibbs:
    def __init__(self, data, K, dimension):
        self.maxIte = 100
        self.data = data
        self.K = K
        self.D = dimension
        self.N = len(data)
        self.label = np.full(shape=self.N, fill_value=-1)
        self.alpha = np.full(shape=self.K, fill_value=1)
        self.pi = np.full(shape=self.K, fill_value=1 / K)
        self.Nk = np.full(shape=self.K, fill_value=0)
        self.mu = np.zeros((self.K, self.D))
        self.sigma = [np.eye(self.D)] * self.K
        self.Zn = [np.full(shape=self.K, fill_value=1 / self.K)] * self.N
        self.psi = np.zeros((self.K, self.D))
        self.dof = np.zeros(3)
        self.mu0 = np.zeros((self.K, self.D))
        self.sigma0 = [np.eye(self.D)] * self.K
        self.dof0 = self.K + self.D
        self.psi0 = np.full(shape=self.D, fill_value=1)
        self.mu00 = np.mean(self.data, axis=0)
        self.sigma00 = np.eye(self.D) * self.mu00

    def initialize(self):
        for k in range(self.K):
            self.mu0[k] = np.mean(self.data, axis=0)
            self.sigma0[k] = invwishart(df=self.dof0, scale=self.psi0).rvs()
            self.mu[k] = multivariate_normal(mean=self.mu0[k], cov=self.sigma0[k]).rvs()

    # Number of elements in each cluster
    def recalculate_Nk(self):
        self.Nk = np.full(shape=self.K, fill_value=0)
        for k in range(self.K):
            for i in range(self.N):
                if self.label[i] == k:
                    self.Nk[k] += 1

    # p(pi|.) = p(pi|z) = Dirichlet(alpha + nk)
    def sample_pi(self):
        self.recalculate_Nk()
        self.pi = np.random.dirichlet(alpha=self.alpha + self.Nk)

    # p(z|.) = [pi_k * MVN(x_n | mu_k, sigma_k)] / [sum[1:K](pi_i * MVN(x_n | mu_i, sigma_i))]
    def sample_z(self):
        zn = np.zeros((self.N, self.K))
        for k in range(self.K):
            normal = multivariate_normal(mean=self.mu[k], cov=self.sigma[k])
            zn[:, k] = normal.pdf(self.data)
        numerator = zn * self.pi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        self.Zn = numerator / denominator
        self.label = np.argmax(self.Zn, axis=1)

    # p(sigma|.) = Inverse_Wishart(psi,dof)
    def sample_sigma(self):
        self.recalculate_Nk()
        psi = np.zeros((self.K, self.D))
        for k in range(self.K):
            for i in range(self.N):
                if self.label[i] == k:
                    psi[k] += np.dot(self.data[i] - self.mu[k], (self.data[i] - self.mu[k]).T)
        self.psi = self.psi0 + psi
        self.dof = self.dof0 + self.Nk
        for k in range(self.K):
            iw = invwishart(df=self.dof[k], scale=self.psi[k])
            self.sigma[k] = iw.rvs()

    # p(mu|.) = Normal(mu0, sigma0)
    def sample_mu(self):
        for k in range(self.K):
            self.sigma0[k] = np.linalg.inv(np.linalg.inv(self.sigma00) + self.Nk[k] * np.linalg.inv(self.sigma[k]))
            x_k = np.sum(self.data[self.label == k], axis=0)
            temp = np.dot(np.linalg.inv(self.sigma[k]), x_k) + np.dot(np.linalg.inv(self.sigma00), self.mu00)
            self.mu0[k] = np.dot(self.sigma0[k], temp)
            normal = multivariate_normal(mean=self.mu0[k], cov=self.sigma0[k])
            self.mu[k] = normal.rvs()

    def main(self):
        self.initialize()
        for i in range(self.maxIte):
            self.sample_pi()
            self.sample_z()
            self.sample_sigma()
            self.sample_mu()
            self.label = np.argmax(self.Zn, axis=1)


if __name__ == '__main__':
    # Data testing Index from 0:7
    data, dataname, datasize, K = df.GetData(2)
    gmm = GMM_Gibbs(data, K, 2)
    gmm.main()
    gp.GMM_data_display(data, gmm.label, gmm.mu, gmm.sigma, K, dataname)