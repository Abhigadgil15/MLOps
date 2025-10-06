from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

def load_data():
    wine = load_wine()
    X = wine.data
    y = wine.target
    return X, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=12)