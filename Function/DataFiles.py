import numpy as np


def Dataset1():
    N = 50
    K = 3
    D = 2
    means = [[10, 2], [0, 0], [5, 9]]
    cov = [[[2, 5], [1, 2]], [[3, 1], [4, 7]], [[1, 8], [9, 2]]]
    X0 = np.random.multivariate_normal(means[0], cov[0], N)
    X1 = np.random.multivariate_normal(means[1], cov[1], 2 * N)
    X2 = np.random.multivariate_normal(means[2], cov[2], 3 * N)
    data = np.concatenate((X0, X1, X2), axis=0)
    label = np.asarray([0] * N + [1] * 2 * N + [2] * 3 * N)
    return data, K, D, label


# --------------------------------------------------------------------------------------
# Read data
# [0]link, [1]name, [2]size, [3]K-cluster
def GetData(index):
    all_link = [
        ['data/shape_sets/Aggregation.txt', 'Aggregation', 788, 7],
        ['data/shape_sets/Compound.txt', 'Compound', 399, 6],
        ['data/shape_sets/D31.txt', 'D31', 3100, 31],
        ['data/shape_sets/flame.txt', 'Flame', 240, 2],
        ['data/shape_sets/jain.txt', 'Jain', 373, 2],
        ['data/shape_sets/pathbased.txt', 'Pathbased', 300, 3],
        ['data/shape_sets/R15.txt', 'R15', 600, 15],
        ['data/shape_sets/Spiral.txt', 'Spiral', 312, 3],
    ]

    link = all_link[index][0]
    dataname = all_link[index][1]
    datasize = all_link[index][2]
    K = all_link[index][3]
    data = []
    label = []
    file = open(link)
    text = file.read().splitlines()
    for temp in text:
        val = list(filter(None, temp.split('\t')))
        data += [[float(val[0]), float(val[1])]]
        label += [int(val[2])]
    file.close()
    data = np.asarray(data)
    return data, dataname, datasize, K