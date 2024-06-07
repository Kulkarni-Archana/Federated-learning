import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from typing import List

def load_dataset(client_id: int):
    df = pd.read_csv('data.csv')

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split the dataset evenly into thirds, removing the remainders
    np.random.seed(42)
    random_choose = np.random.choice(X.index, (len(X) % 3), replace=False)
    X = X.drop(random_choose)
    y = y.drop(random_choose)

    # Split the dataset into 3 subsets for 3 clients
    X_split, y_split = np.split(X, 3), np.split(y, 3)
    X1, y1 = X_split[0], y_split[0]
    X2, y2 = X_split[1], y_split[1]
    X3, y3 = X_split[2], y_split[2]

    # Split the training set and testing set in 80% ratio
    X_train, y_train, X_test, y_test = [], [], [], []
    train_size = 0.8

    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size=train_size, random_state=42)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, train_size=train_size, random_state=42)
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, train_size=train_size, random_state=42)

    X_train.append(X1_train)
    X_train.append(X2_train)
    X_train.append(X3_train)

    y_train.append(y1_train)
    y_train.append(y2_train)
    y_train.append(y3_train)

    X_test.append(X1_test)
    X_test.append(X2_test)
    X_test.append(X3_test)

    y_test.append(y1_test)
    y_test.append(y2_test)
    y_test.append(y3_test)

    return X_train[client_id], y_train[client_id], X_test[client_id], y_test[client_id]

def get_params(model: MLPClassifier) -> List[np.ndarray]:
    return [coef for coef in model.coefs_] + [intercept for intercept in model.intercepts_]

def set_params(model: MLPClassifier, params: List[np.ndarray]) -> None:
    model.coefs_ = params[:len(model.coefs_)]
    model.intercepts_ = params[len(model.coefs_):]

