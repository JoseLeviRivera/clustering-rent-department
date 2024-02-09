from sklearn.cluster import AgglomerativeClustering


def run_model_training(n_clusters, X):
    # Aplicación de algoritmos de clustering: Método jerárquico
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    clusters_agglomerative = agglomerative.fit_predict(X)
    return clusters_agglomerative
