import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.spatial.distance as dt
import scipy.stats as stats

group_means = np.array([[+0.0, +5.5],
                        [+0.0, +0.0],
                        [+0.0, -5.5]])

group_covariances = np.array([[[+6.0, +0.0],
                               [+0.0, +0.4]],
                              [[+3.2, +2.8],
                               [+2.8, +3.2]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]]])

# read data into memory
data_set = np.genfromtxt("hw06_data_set.csv", delimiter = ",")

# get X values
X = data_set[:, [0, 1]]

# set number of clusters
K = 3

# STEP 2
# should return initial parameter estimates
# as described in the homework description
def initialize_parameters(X, K):
    # your implementation starts below
    centroids = np.genfromtxt("hw06_initial_centroids.csv", delimiter = ",")
    dist = dt.cdist(X, centroids)
    c_labels = np.argmin(dist, axis=1)

    means = np.array([np.mean(X[c_labels == k], axis=0) for k in range(K)])

    covariances = np.array([np.cov(X[c_labels == k].T) for k in range(K)])

    priors = np.array([np.mean(c_labels == k) for k in range(K)])
    # your implementation ends above
    return(means, covariances, priors)

means, covariances, priors = initialize_parameters(X, K)

# STEP 3
# should return final parameter estimates of
# EM clustering algorithm
def em_clustering_algorithm(X, K, means, covariances, priors):
    # your implementation starts below
    iterations = 100
    n_samples, n_features = X.shape

    for i in range(iterations):

        resp = np.zeros((n_samples, K))
        for k in range(K):
            resp[:, k] = priors[k] * stats.multivariate_normal.pdf(X, mean=means[k], cov=covariances[k])
        resp /= np.sum(resp, axis=1, keepdims=True)

        for k in range(K):
            sum_k = np.sum(resp[:, k])
            means[k] = np.sum(resp[:, k].reshape(-1, 1) * X, axis=0) / sum_k
            covariances[k] = np.dot((resp[:, k].reshape(-1, 1) * (X - means[k])).T, (X - means[k])) / sum_k
            priors[k] = sum_k / n_samples

    assignments = np.argmax(resp, axis=1)
    # your implementation ends above
    return(means, covariances, priors, assignments)

means, covariances, priors, assignments = em_clustering_algorithm(X, K, means, covariances, priors)
print(means)
print(priors)

# STEP 4
# should draw EM clustering results as described
# in the homework description
def draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments):
    # your implementation starts below
    plt.figure(figsize=(8, 6))

    # Scatter plot of data points with color-coded clusters
    colors = ['blue', 'green', 'red']
    for k in range(K) :
        c_points = X[assignments == k]
        plt.scatter(c_points[:, 0], c_points[:, 1], color=colors[k], alpha=0.5, label=f'Cluster {k + 1}')

    # Draw original Gaussian densities
    x, y = np.meshgrid(np.linspace(-8.2, 8.2, 200), np.linspace(-8.2, 8.2, 200))
    for k in range(K) :
        org_density = stats.multivariate_normal.pdf(np.dstack((x, y)), mean=group_means[k], cov=group_covariances[k])
        plt.contour(x, y, org_density, levels=[0.01], colors='black', linestyles='dashed', linewidths=2)

    # Draw Gaussian densities estimated by EM algorithm
    for k in range(K) :
        est_density = stats.multivariate_normal.pdf(np.dstack((x, y)), mean=means[k], cov=covariances[k])
        plt.contour(x, y, est_density, levels=[0.01], colors=colors[k], linestyles='solid', linewidths=2)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

    # your implementation ends above
    
draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments)

