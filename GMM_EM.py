# ------------------------------------------------------------
# Gaussian Mixture Model - Expectation Maximization
# ------------------------------------------------------------


import numpy as np
from scipy.stats import multivariate_normal
from Function import DataFiles as df, GraphDisplay as gp


class GMM_EM:
    def __init__(self, data, K, dimension):
        self.data = data
        self.K = K
        self.D = dimension
        self.N = len(data)
        self.max_ite = 1000
        self.epsilon = 0.0001
        self.pi = np.full(shape=self.K, fill_value=1 / self.K)
        self.response = [np.full(shape=self.K, fill_value=1 / self.K)] * self.N
        self.mu = self.data[np.random.choice(self.N, self.K, False), :]
        self.sigma = [np.eye(self.D)] * self.K
        self.logLikelihood = []
        self.label = []

    # E step: evaluate the responsibilities
    def expectation(self):
        likelihood = np.zeros((self.N, self.K))
        for k in range(self.K):
            distribution = multivariate_normal(mean=self.mu[k], cov=self.sigma[k])
            likelihood[:, k] = distribution.pdf(self.data)
        numerator = likelihood * self.pi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        self.response = numerator / denominator
        log_likelihood = np.sum(np.log(np.sum(likelihood, axis=1)))
        self.logLikelihood.append(log_likelihood)

    # M step: re-estimate parameters: mu_k , sigma_k , pi_k
    def maximization(self):
        for k in range(self.K):
            response = self.response[:, [k]]
            total = response.sum()
            self.mu[k] = (self.data * response).sum(axis=0) / total
            self.sigma[k] = np.cov(self.data.T, aweights=(response / total).flatten(), bias=True)
        self.pi = self.response.mean(axis=0)

    def is_converged(self):
        if len(self.logLikelihood) < 2:
            return 0
        eps = np.abs(self.logLikelihood[-1] - self.logLikelihood[-2])
        if self.epsilon > eps:
            return 1
        return 0

    def main(self):
        for i in range(self.max_ite):
            self.expectation()
            self.maximization()
            self.label = np.argmax(self.response, axis=1)
            if self.is_converged():
                break
        self.label = np.argmax(self.response, axis=1)


''''''

if __name__ == '__main__':
    # Data testing Index from 0:7
    data, dataname, datasize, K = df.GetData(3)
    gmm = GMM_EM(data, K, 2)
    gmm.main()
    gp.GMM_data_display(data, gmm.label, gmm.mu, gmm.sigma, K, dataname)


''' Using sklearn Lib
    from sklearn.mixture import GaussianMixture
    gmm_sk = GaussianMixture(K)
    result_sk = gmm_sk.fit(data)
    print(result_sk.means_, result_sk.covariances_)
'''