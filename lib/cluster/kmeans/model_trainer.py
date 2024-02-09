from sklearn.cluster import KMeans


def run_model_training(n_clusters, X):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters_kmeans = kmeans.fit_predict(X)
    return clusters_kmeans