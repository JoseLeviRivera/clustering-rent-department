import pandas as pd


def read_csv(file):
    return pd.read_csv(file, delimiter=';', encoding='ISO-8859-1')
