from sklearn.neighbors import KNeighborsClassifier


"""
Esta función realiza el entrenamiento de un clasificador K Vecinos Más Cercanos (KNN) con el número de vecinos
especificado (n_neighbors) utilizando los datos de entrenamiento proporcionados (X_train, y_train). Luego, evalúa
el modelo entrenado en los datos de prueba (X_test, y_test) y genera un informe de clasificación. Los resultado
del entrenamiento y la evaluación del modelo se guardan en un archivo JSON especificado por el parámetro save_path.
Parámetros:
    - n_neighbors (int): Número de vecinos a considerar en el algoritmo KNN.
    - X_train (similar a matriz): La matriz de características de los datos de entrenamiento.
    - X_test (similar a matriz): La matriz de características de los datos de prueba.
    - y_train (similar a matriz): Las etiquetas de destino de los datos de entrenamiento.
    - y_test (similar a matriz): Las etiquetas de destino de los datos de prueba.
    - save_path (str, opcional): La ruta del archivo para guardar los resultados de la evaluación del modelo en formato JSON.
    El valor predeterminado es "model_results.json".
Devoluciones:
    knn (KNeighborsClassifier): El clasificador K Vecinos Más Cercanos entrenado
"""
def run_model_training(n_neighbors, X_train, X_test, y_train, y_test, save_path="model_results.json"):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    knn.score(X_test, y_test)
    return knn
