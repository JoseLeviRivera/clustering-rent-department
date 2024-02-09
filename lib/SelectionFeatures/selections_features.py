from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split


def generate_features_selector(modelo, X, y, n, dir):
    sfs = SequentialFeatureSelector(modelo, n_features_to_select=n, direction=dir)
    sfs.fit(X, y)
    return sfs.get_support()

def create_train_test_data(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return {'x_train': X_train, 'x_test': X_test,
            'y_train': y_train, 'y_test': y_test}