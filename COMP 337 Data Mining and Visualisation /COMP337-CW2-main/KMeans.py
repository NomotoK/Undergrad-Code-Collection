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



def initial_selection(X, k):
    """
    Select initial cluster centroids randomly.

    Parameters: X (numpy.ndarray): The data matrix.

    Returns: centroids (numpy.ndarray): The initial cluster centroids.
    """
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]



def compute_distance(a, b):
    """
    Compute Euclidean distance between two numpy arrays.

    Parameters: a (numpy.ndarray): The first array. b (numpy.ndarray): The second array.
                
    Returns: distance (float): The Euclidean distance between a and b.
    """
    return np.linalg.norm(a - b)



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
    Recompute the cluster centroids as the mean of assigned points.

    Parameters: X (numpy.ndarray): The data matrix. clusters (numpy.ndarray): The cluster IDs assigned to each data point. k (int): The number of clusters.

    Returns: centroids (numpy.ndarray): The updated cluster centroids.
    """
    return np.array([X[clusters == i].mean(axis=0) for i in range(k)])



def cluster_name(X, k, max_iters=100):
    """
    Run the k-means clustering algorithm and return cluster labels and centroids.

    Parameters: X (numpy.ndarray): The data matrix. k (int): The number of clusters. max_iters (int): The maximum number of iterations.

    Returns: clusters (numpy.ndarray): The cluster IDs assigned to each data point. centroids (numpy.ndarray): The cluster centroids.
    """
    centroids = initial_selection(X, k)
    for _ in range(max_iters): # Repeat until convergence
        clusters = assign_cluster_ids(X, centroids)
        new_centroids = compute_cluster_representatives(X, clusters, k)
        if np.all(centroids == new_centroids):# Check for convergence
            break
        centroids = new_centroids
    return clusters, centroids



def silhouette_score(X, labels, metric='euclidean'):
    """
    Calculate the mean silhouette coefficient for all samples.

    Parameters: X (numpy.ndarray): The data matrix. labels (numpy.ndarray): The cluster IDs assigned to each data point. metric (str): The distance metric to use.

    Returns: s (float): The mean silhouette coefficient.
    """
    from scipy.spatial.distance import pdist, squareform
    
    # Compute the distance matrix
    distances = squareform(pdist(X, metric=metric))
    unique_clusters = np.unique(labels)
    
    # Compute 'a' and 'b' for each sample
    a = np.array([np.mean(distances[i][labels == labels[i]]) for i in range(len(X))])
    b = np.array([np.min([np.mean(distances[i][labels == label])
                          for label in unique_clusters if label != labels[i]]) for i in range(len(X))])
    
    # Calculate the silhouette score
    s = (b - a) / np.maximum(a, b)
    
    # Return the mean silhouette score, excluding NaN values
    return np.nanmean(s)



def plot_silhouette(k_values, silhouette_scores):
    """
    Plot the silhouette scores for each value of k.

    Parameters: k_values (list): The list of k values. silhouette_scores (list): The corresponding silhouette scores.

    Returns: None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.title('Silhouette Scores vs Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig('KMeans_silhouette_scores.png')
    plt.show()



def main(data_path):
    """
    Main function to run the clustering algorithm and plot the silhouette scores.

    Parameters: data_path (str): The path to the dataset file.

    Returns: None
    """
    X = load_data(data_path)
    silhouette_scores = []
    k_range = range(1, 10)
    
    for k in k_range:
        if k == 1:
            silhouette_scores.append(0)  # Silhouette score is not meaningful for k=1
        else:
            clusters, centroids = cluster_name(X, k)# Run the clustering algorithm
            score = silhouette_score(X, clusters)
            silhouette_scores.append(score)
    
    plot_silhouette(k_range, silhouette_scores)



if __name__ == '__main__':
    
    data_path = 'dataset'  
    main(data_path)
