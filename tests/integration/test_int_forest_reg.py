import pytest
import numpy as np
import pandas as pd
from randomforests.utils import _make_dataset
from randomforests import RandomForestRegressor


def test_make_bootsrap():
    """
    Cant really do good test since it has random sample with replacement
    """
    X = np.array([[0.1],
                  [0.5],
                  [0.7],
                  [0.9]])

    y = np.array([0.1, 0.5, 0.7, 0.9])

    dataset = _make_dataset(X,y)

    forest = RandomForestRegressor()
    tree   = forest._bootstrap_tree(dataset=dataset, n_features=1)

    assert tree.n_features == 1

def test_fit():
    """
    Cant really do good test since it has random sample with replacement
    """
    X = np.array([[0.1],
                  [0.5],
                  [0.7],
                  [0.9]])

    y = np.array([0.1, 0.5, 0.7, 0.9])

    forest = RandomForestRegressor()
    model  = forest.fit(X,y)

    assert len(model.trees) == 10

predict_tests = [(np.array([0.1, 0.5, 0.7, 0.9])),
                 (pd.Series([0.1, 0.5, 0.7, 0.9]))]

@pytest.mark.parametrize('y', predict_tests)
def test_predict(y):
    """
    Cant really do good test since it has random sample with replacement

    But check to make sure the shape is consistent and the predicted classes
    with the training set target values.
    """

    X = np.array([[0., 0.1, 0.1],
                  [0., 0.5, 0.5],
                  [0., 0.7, 0.7],
                  [0., 0.9, 0.9]])

    y = np.array([0.1, 0.5, 0.7, 0.9])

    forest = RandomForestRegressor()
    model  = forest.fit(X,y)

    preds  = model.predict(X)
    correct_size  = len(preds) == 4

    bounded_max   = np.max(preds) <= 0.9 + 1e-14 # ehhh
    bounded_min   = np.min(preds) >= 0.1

    assert correct_size and bounded_min and bounded_max