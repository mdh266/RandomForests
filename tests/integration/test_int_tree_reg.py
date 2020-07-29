import pytest
import pandas as pd
import numpy as np
from randomforests.TreeRegressor import DecisionTreeRegressor

def test__get_split():
    tree      = DecisionTreeRegressor(n_features = 1)
    result    = tree._get_split(np.array([[0.1,0.1],[0.5,0.5],[0.7,0.7],[0.9,0.9]]))

    column    = result["column"] == 0
    value     = result["value"]  == 0.5

    left_grp  = np.array_equal(result["groups"][0],np.array([[0.1, 0.1]]))
    right_grp = np.array_equal(result["groups"][1],np.array([[0.5, 0.5],
                                                             [0.7, 0.7],
                                                             [0.9, 0.9]]))

    assert (column and value and left_grp and right_grp)

public_predict_test = [
    (np.array([[0. , 0.1],
               [0. , 0.5],
               [0. , 0.7],
               [0. , 0.9]]), # X
     np.array([0.1, 0.5, 0.7, 0.9]),# y
     np.array([0.1, 0.6, 0.6, 0.9])),# y_test

    (pd.DataFrame({"c1":[0., 0., 0., 0.],
                   "c2":[0.1, 0.5, 0.7, 0.9]}),# X
     pd.Series([0.1, 0.5, 0.7, 0.9]),# y
     np.array([0.1, 0.6, 0.6, 0.9])# y_test
     )# y_test
]


@pytest.mark.parametrize('X, y, y_test', public_predict_test)
def test_predict(X, y, y_test):
    tree   = DecisionTreeRegressor()
    model  = tree.fit(X, y)
    assert np.array_equal(model.predict(X), y_test)


def test_score():
    X = pd.DataFrame({"c1":[0., 0., 0., 0.],
                      "c2":[0.1, 0.5, 0.7, 0.9]})

    y = pd.Series([0.1, 0.5, 0.7, 0.9])

    tree   = DecisionTreeRegressor()
    model  = tree.fit(X, y)

    assert model.score(X,y) == pytest.approx(0.004999999)