import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Setting the seed for reproducibility
random.seed(42)
np.random.seed(42)


def load_data(filepath):
    """
    Load and format the data for clustering.

    Parameters: filepath (str): The path to the dataset file.

    Returns: X (numpy.ndarray): The data matrix.
    """
    data = pd.read_csv(filepath, delimiter=' ')
    X = data.iloc[:, 1:].values
    return X



def compute_distance(a, b):
    """
    Compute Euclidean distance between two numpy arrays.

    Parameters: a (numpy.ndarray): The first array. b (numpy.ndarray): The second array.

    Returns: distance (float): The Euclidean distance between a and b.
    """
    return np.linalg.norm(a - b)



def initial_selection(X, k):
    """
    Select initial cluster centroids randomly.

    Parameters: X (numpy.ndarray): The data matrix.

    Returns: centroids (numpy.ndarray): The initial cluster centroids.
    """
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]



def assign_cluster_ids(X, centroids):
    """
    Assign each data point to the closest cluster centroid.

    Parameters: X (numpy.ndarray): The data matrix. centroids (numpy.ndarray): The cluster centroids.

    Returns: cluster_ids (numpy.ndarray): The cluster IDs assigned to each data point.
    """
    distances = np.array([[compute_distance(x, centroid) for centroid in centroids] for x in X])
    return np.argmin(distances, axis=1)



def compute_cluster_representatives(X, clusters, k):
    """
    Compute new centroids as the mean of points in each cluster.

    Parameters: X (numpy.ndarray): The data matrix. clusters (numpy.ndarray): The cluster IDs for each data point. k (int): The number of clusters.

    Returns: centroids (numpy.ndarray): The new cluster centroids.
    """
    return np.array([X[clusters == i].mean(axis=0) for i in range(k) if len(X[clusters == i]) > 0])



def k_means(X, k, max_iters=10):
    """
    Run the k-means clustering algorithm and return cluster labels and centroids.

    Parameters: X (numpy.ndarray): The data matrix. k (int): The number of clusters. max_iters (int): The maximum number of iterations.

    Returns: clusters (numpy.ndarray): The cluster IDs assigned to each data point. centroids (numpy.ndarray): The cluster centroids.
    """
    centroids = initial_selection(X, k)
    for _ in range(max_iters):
        clusters = assign_cluster_ids(X, centroids)
        new_centroids = compute_cluster_representatives(X, clusters, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids



def bisecting_k_means(X, num_clusters):
    """
    Bisecting k-means clustering algorithm.

    Parameters: X (numpy.ndarray): The data matrix. num_clusters (int): The number of clusters.

    Returns: clusters (numpy.ndarray): The cluster IDs assigned to each data point.
    """
    clusters = np.zeros(X.shape[0], dtype=int)
    current_clusters = [X]
    
    while len(current_clusters) < num_clusters:
        largest_cluster_idx = np.argmax([len(cluster) for cluster in current_clusters])
        cluster_to_split = current_clusters.pop(largest_cluster_idx)
        
        if len(cluster_to_split) < 2:
            current_clusters.append(cluster_to_split)
            continue
        
        sub_clusters, _ = k_means(cluster_to_split, 2)
        
        current_clusters.extend([cluster_to_split[sub_clusters == i] for i in range(2) if len(cluster_to_split[sub_clusters == i]) > 0])
    
    current_cluster_id = 0
    for cluster in current_clusters:
        indices = np.isin(X, cluster).all(axis=1)
        clusters[indices] = current_cluster_id
        current_cluster_id += 1
        
    return clusters



def silhouette_score(X, labels):
    """
    Compute the silhouette score for a clustering.

    Parameters: X (numpy.ndarray): The data matrix. labels (numpy.ndarray): The cluster IDs assigned to each data point.

    Returns: s (float): The silhouette score.
    """
    # Compute pairwise distances
    distances = np.array([[compute_distance(x, y) for y in X] for x in X])
    unique_clusters = np.unique(labels)
    
    if len(unique_clusters) == 1:
        return 0  # Can't compute silhouette for a single cluster
    
    a = np.array([np.mean(distances[i][labels == labels[i]]) for i in range(len(X))])
    b = np.array([np.min([np.mean(distances[i][labels == label])
                          for label in unique_clusters if label != labels[i]]) for i in range(len(X))])
    
    s = (b - a) / np.maximum(a, b)
    return np.nanmean(s)



def plot_silhouette(k_values, silhouette_scores):
    """
    Plot the silhouette scores for different numbers of clusters.

    Parameters: k_values (list): The list of k values. silhouette_scores (list): The corresponding silhouette scores.

    Returns: None
    """

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.title('Silhouette Scores vs Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig('BisectingKMeans_silhouette_scores.png')
    plt.show()



def main(data_path):
    """
    Load the data and plot the silhouette scores for different numbers of clusters.

    Parameters: data_path (str): The path to the dataset file.

    Returns: None
    """
    X = load_data(data_path)
    silhouette_scores = []
    
    for num_clusters in range(1, 10):
        clusters = bisecting_k_means(X, num_clusters)
        score = silhouette_score(X, clusters)
        silhouette_scores.append(score)
    
    plot_silhouette(range(1, 10), silhouette_scores)



if __name__ == '__main__':
    data_path = 'dataset'  
    main(data_path)
