import pytest
# TODO: add necessary import
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import (train_model, inference, compute_model_metrics,)

test_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@pytest.fixture(scope="module")
def test_data():
    """
    # load 10,000 rows of data from census.csv
    """
    test_data_path = os.path.join("data", "census.csv")
    test_data = pd.read_csv(test_data_path)
    return test_data.sample(10000)

@pytest.fixture(scope="module")
def create_test_model(test_data):
    x, y, encoder, lb = process_data(
        test_data,
        categorical_features = test_features,
        label = "salary",
        training = True,
    )
    test_model = train_model(x, y)
    return x, y, test_model

# TODO: implement the first test. Change the function name and input as needed
def test_one(test_data):
    """
    # verify that the train and test datasets are an 80/20 split
    """
    # Your code here
    train, test = train_test_split(test_data, test_size = 0.20)
    
    df_size = len(test_data)
    num_test_rows = int(df_size * 0.20)
    num_training_rows = df_size - num_test_rows

    assert len(train) == num_training_rows
    assert len(test) == num_test_rows
    assert len(train) + len(test) == df_size

    pass


# TODO: implement the second test. Change the function name and input as needed
def test_two(test_data):
    """
    # verify that the saved model uses random forest classifier
    """
    # Your code here

    test_model_path = os.path.join("model", "model.pkl")

    with open(test_model_path, "rb") as f:
        saved_model = pickle.load(f)
        
    assert isinstance(saved_model, RandomForestClassifier)

    pass


# TODO: implement the third test. Change the function name and input as needed
def test_three(create_test_model):
    """
    # verify that compute_model_metrics returns known precision, recall and F1 values.
    # distinct values are used to also check for proper return order.
    """
    # Your code here

    y_true = np.array([1, 1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0])

    expected_precision = 2 / 3
    expected_recall = 0.5
    expected_fbeta = 4 / 7

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert precision == pytest.approx(expected_precision, rel = 1e-6)
    assert recall == pytest.approx(expected_recall, rel = 1e-6)
    assert fbeta == pytest.approx(expected_fbeta, rel = 1e-6)
    
    pass
