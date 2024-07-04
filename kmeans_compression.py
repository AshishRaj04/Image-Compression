import matplotlib.pyplot as plt
import numpy as np

def nearest_centroid(X , centroid):
  k = centroid.shape[0]
  c = np.zeros(X.shape[0] , dtype=int)

  for i in range(X.shape[0]):
    distance = []
    for j in range(k):
      norm_ij = np.linalg.norm(X[i] - centroid[j])
      distance.append(norm_ij)
    c[i] = np.argmin(distance)
  return c

def compute_centroid(X , c , k):
  m , n = X.shape
  centroids = np.zeros((k , n))
  for i in range(k):
    centroids[i] = np.mean(X[c == i] , axis = 0)
  return centroids

def initialize_centroid(X , k):
  random_index = np.random.permutation(X.shape[0])
  return X[random_index[:k]]


def run_kMeans(X, initial_centroids, iter):

    m, n = X.shape
    k = initial_centroids.shape[0]
    centroids = initial_centroids
    idx = np.zeros(m)

    for i in range(iter):

        print("K-Means iteration %d/%d" % (i, iter-1))

        idx = nearest_centroid(X, centroids)

        centroids = compute_centroid(X, idx, k)
    
    return centroids, idx

def startCompressing(X_img):
    k = 16
    iter = 10
    initial_centroids = initialize_centroid(X_img, k)
    centroids, idx = run_kMeans(X_img, initial_centroids, iter)
    
    c = nearest_centroid(X_img , centroids)
    X_recovered = centroids[c, :]
    
    return X_recovered

