import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Leer el archivo CSV
data = pd.read_csv("lib/data/apartments_for_rent_classified_10K.csv", delimiter=';', encoding='ISO-8859-1')

# Seleccionar características
features = ["category", "bathrooms", "bedrooms", "fee", "pets_allowed", "price", "square_feet",
            "cityname", "state", "latitude", "longitude", "time"]
# Eliminar filas con valores nulos
data = data.dropna()

# Antes de las líneas que generan las advertencias
pd.set_option('future.no_silent_downcasting', True)

# Mapear la columna 'category' a valores numéricos
mapeo_category = {'housing/rent/apartment': 1}
data['category'] = data['category'].replace(mapeo_category)


# Mapear la columna 'fee' a valores numéricos
mapeo_fee = {'No': 2, 'Yes': 3}
data['fee'] = data['fee'].replace(mapeo_fee)

# Mapear la columna 'pets_allowed' a valores numéricos
mapeo_pets = {'Cats,Dogs': 4, 'Cats': 5, 'Dogs': 6}
data['pets_allowed'] = data['pets_allowed'].replace(mapeo_pets)

# Mapear la columna 'state' a valores numéricos
mapeo_estado = {'NC': 1, 'WA': 2, 'IL': 3, 'TN': 4, 'CA': 5, 'AK': 6, 'IA': 7, 'KY': 8, 'WI': 9, 'DC': 10, 'IN': 11,
                'FL': 12, 'OR': 13, 'MN': 14, 'MD': 15, 'TX': 16, 'CO': 17, 'NV': 18, 'CT': 19, 'OH': 20, 'MO': 21,
                'AZ': 22, 'MA': 23, 'PA': 24, 'NJ': 25, 'OK': 26, 'MI': 27, 'RI': 28, 'NH': 29, 'VA': 30, 'NE': 31,
                'LA': 32, 'ND': 33, 'AL': 34, 'AR': 35, 'KS': 36, 'NY': 37, 'UT': 38, 'GA': 39, 'VT': 40, 'SC': 41,
                'NM': 42, 'ID': 43, 'SD': 44, 'HI': 45, 'WV': 46, 'MS': 47, 'DE': 48, 'ME': 49, 'MT': 50, 'WY': 51}
data['state'] = data['state'].replace(mapeo_estado)

# Mapear la columna 'cityname' a valores numéricos
ciudades_unicas = data['cityname'].unique()
mapeo_ciudades = {ciudad: i + 1 for i, ciudad in enumerate(ciudades_unicas)}
data['cityname'] = data['cityname'].replace(mapeo_ciudades)

X = data[features]

# Visualización de los datos
pd.plotting.scatter_matrix(X, figsize=(12, 12))
plt.show()

# Normalización de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Clustering con K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_clusters = kmeans.fit_predict(X_scaled)

# Obtener los centroides
centroids = kmeans.cluster_centers_
print(centroids)


# Clustering jerárquico
agglomerative = AgglomerativeClustering(n_clusters=3)
agglomerative_clusters = agglomerative.fit_predict(X_scaled)

# Visualización de los clusters en un gráfico de dispersión
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_clusters, cmap='viridis')
plt.title('Clustering con K-Means')

plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=agglomerative_clusters, cmap='viridis')
plt.title('Clustering jerárquico')
plt.show()

# Visualización del clustering por K-Means
plt.scatter(X['square_feet'], X['price'], c=kmeans_clusters, cmap='viridis')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.title('Clustering de apartamentos para alquilar (K-Means)')
plt.show()

# Visualización del clustering por Método Jerárquico
plt.scatter(X['square_feet'], X['price'], c=agglomerative_clusters, cmap='viridis')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.title('Clustering de apartamentos para alquilar (Método Jerárquico)')
plt.show()

# Dendrograma para el clustering jerárquico
plt.figure(figsize=(12, 6))
linkage_data = linkage(X_scaled, method='ward', metric='euclidean')
dendrogram(linkage_data)
plt.title('Dendrograma de Clustering Jerárquico')
plt.xlabel('Índices de las muestras')
plt.ylabel('Distancias')
plt.show()

# Análisis de los clusters
data['KMeans_Cluster'] = kmeans_clusters
data['Agglomerative_Cluster'] = agglomerative_clusters


# Imprimir el análisis de los clusters generados por K-Means
print("Análisis de los clusters generados por K-Means:")
print(data.groupby('KMeans_Cluster')[features].mean())

# Imprimir el análisis de los clusters generados por Clustering Jerárquico
print("\nAnálisis de los clusters generados por Clustering Jerárquico:")
print(data.groupby('Agglomerative_Cluster')[features].mean())