# ------------------------------------------------------------
# K-Means Clustering
# ------------------------------------------------------------


import numpy as np
from scipy.spatial.distance import cdist
from Function import DataFiles as df, GraphDisplay as gp


class KMeans:
    def __init__(self, data, K):
        self.data = data
        self.K = K

    def init_centers(self):
        rd = np.random.choice(self.data.shape[0], self.K, replace=False)
        _centers = [self.data[row_index, :] for row_index in rd]
        return np.asarray(_centers)

    def assign_labels(self, _centers):
        distance = cdist(self.data, _centers)
        index = np.argmin(distance, axis=1)
        return index

    def update_centers(self, _label):
        _centers = []
        _X = []
        for k in range(self.K):
            _X.append([])
        for i in range(len(self.data)):
            for k in range(self.K):
                if _label[i] == k:
                    _X[k].append(data[i])
        for k in range(self.K):
            _centers.append(np.mean(_X[k], axis=0))
        return np.asarray(_centers)

    def is_converged(self, _centers, _new_centers):
        a = set([tuple(a) for a in _centers])
        b = set([tuple(a) for a in _new_centers])
        if a == b:
            return 1
        return 0

    def main(self):
        _centers = self.init_centers()
        i = 0
        while 1:
            _label = self.assign_labels(_centers)
            new_centers = self.update_centers(_label)
            if self.is_converged(_centers, new_centers):
                break
            _centers = new_centers
            i += 1
        return _centers, _label, i


''''''

if __name__ == '__main__':
    # Data testing Index from 0:7
    data, dataname, datasize, K = df.GetData(0)
    kmeans = KMeans(data, K)
    centers, labels, it = kmeans.main()
    gp.data_display(data, K, labels, dataname)

''' Using sklearn Lib
    from sklearn.cluster import kmeans_plusplus
    skikmeans, index = kmeans_plusplus(data, n_clusters=3, random_state=0)
    print(skikmeans)
    data_display(data, labels, skikmeans)
'''