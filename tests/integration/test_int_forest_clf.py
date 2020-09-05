import pytest
import numpy as np
import pandas as pd
from randomforests.utils import _make_dataset
from randomforests import RandomForestClassifier


def test_make_bootsrap():
    """
    Cant really do good test since it has random sample with replacement
    """
    X = np.array([[0.1],
                  [0.5],
                  [0.7],
                  [0.9]])

    y = np.array([0, 0, 1, 1])

    dataset = _make_dataset(X,y)

    forest = RandomForestClassifier()
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

    y = np.array([0, 0, 1, 1])


    forest = RandomForestClassifier()
    model  = forest.fit(X,y)

    assert len(model.trees) == 10


predict_tests = [(np.array([0, 0, 1, 1])),
                 (pd.Series([0, 0, 1, 1]))]

@pytest.mark.parametrize('y', predict_tests)
def test_predict(y):
    """
    Cant really do good test since it has random sample with replacement

    But check to make sure the shape is consistent and the predicted values
    are with in the training set range.
    """
    X = np.array([[0.1],
                  [0.5],
                  [0.7],
                  [0.9]])

    forest = RandomForestClassifier()
    model  = forest.fit(X,y)

    correct_size  = len(model.predict(X)) == 4
    correct_class = np.array_equal(np.unique(model.predict(X)), np.array([0,1]))

    assert correct_size and correct_class
