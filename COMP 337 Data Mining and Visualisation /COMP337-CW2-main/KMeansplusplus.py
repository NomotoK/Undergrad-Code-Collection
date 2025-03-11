import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


# Setting a random seed for reproducibility
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
    """Compute the Euclidean distance between two numpy arrays."""
    return np.sqrt(np.sum((a - b) ** 2))

def initial_selection_plus(X, k):
    """
    Choose initial centroids using the k-means++ algorithm.

    Parameters: X (numpy.ndarray): The data matrix. k (int): The number of clusters.

    Returns: centroids (numpy.ndarray): The initial cluster centroids.
    """
    n = X.shape[0]
    centroids = [X[np.random.randint(0, n)]]
    for _ in range(1, k):
        distances = np.array([min([compute_distance(x, centroid) for centroid in centroids]) for x in X])
        probabilities = distances ** 2
        probabilities /= probabilities.sum()
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        i = np.searchsorted(cumulative_probabilities, r)
        centroids.append(X[i])
    return np.array(centroids)



def assign_cluster_ids(X, centroids):
    """
    Assign each data point to the nearest centroid.

    Parameters: X (numpy.ndarray): The data matrix. centroids (numpy.ndarray): The cluster centroids.

    Returns: clusters (numpy.ndarray): The cluster IDs for each data point.
    """
    distances = np.array([[compute_distance(x, centroid) for centroid in centroids] for x in X])
    return np.argmin(distances, axis=1)



def compute_cluster_representatives(X, clusters, k):
    """
    Compute new centroids as the mean of points in each cluster.
    
    Parameters: X (numpy.ndarray): The data matrix. clusters (numpy.ndarray): The cluster IDs for each data point. k (int): The number of clusters.

    Returns: centroids (numpy.ndarray): The new cluster centroids.
    """
    return np.array([X[clusters == i].mean(axis=0) for i in range(k)])



def compute_silhouette_score(X, labels):
    """
    Compute the silhouette score manually.
    
    Parameters: X (numpy.ndarray): The data matrix. clusters (numpy.ndarray): The cluster IDs for each data point.

    Returns: s (float): The silhouette score.
    """
    n = X.shape[0]
    a = np.zeros(n)
    b = np.inf * np.ones(n)
    
    for i in range(n):
        same_cluster = (labels == labels[i])
        other_cluster = (labels != labels[i])
        
        a[i] = np.mean([compute_distance(X[i], X[j]) for j in np.where(same_cluster)[0] if i != j])
        
        for label in set(labels):
            if label != labels[i]:
                b[i] = min(b[i], np.mean([compute_distance(X[i], X[j]) for j in np.where(labels == label)[0]]))
    
    s = (b - a) / np.maximum(a, b)
    return np.nanmean(s)



def k_means_plus(X, k, max_iters=100):
    """
    K-means algorithm using k-means++ for initialization.

    Parameters: X (numpy.ndarray): The data matrix. k (int): The number of clusters. max_iters (int): The maximum number of iterations.

    Returns: clusters (numpy.ndarray): The cluster IDs for each data point. centroids (numpy.ndarray): The cluster centroids.
    """
    centroids = initial_selection_plus(X, k)
    for _ in range(max_iters):
        clusters = assign_cluster_ids(X, centroids)
        new_centroids = compute_cluster_representatives(X, clusters, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids



def plot_silhouette(k_range, scores):
    """
    Plot silhouette scores.

    Parameters: k_range (range): The range of k values. scores (list): The silhouette scores.

    Returns: None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, scores, marker='o')
    plt.title('Silhouette Scores vs Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig('KMeansplusplus_silhouette_scores.png')
    plt.show()



def main(data_path):
    """
    Main function to run the k-means++ algorithm.

    Parameters: data_path (str): The path to the dataset file.

    Returns: None
    """
    X = load_data(data_path)
    k_range = range(1, 10)
    scores = []
    
    for k in k_range:
        if k == 1:
            scores.append(0)  # Silhouette score is not defined for k=1
        else:
            clusters, centroids = k_means_plus(X, k)
            score = compute_silhouette_score(X, clusters)
            scores.append(score)
    
    plot_silhouette(k_range, scores)



if __name__ == '__main__':
    data_path = 'dataset'
    main(data_path)
