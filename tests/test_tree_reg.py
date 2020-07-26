import pytest
import pandas as pd
import numpy as np
from randomforests.TreeRegressor import DecisionTreeRegressor


leaf_tests =[(np.array([[0.],[0.], [1.],[0.]]),0.25),
             (np.array([[0.],[1.], [1.],[1.]]),0.75),
             (np.array([[1.],[1.], [1.],[1.]]),1.0)]


@pytest.mark.parametrize('y, expected', leaf_tests)
def test_make_leaf(y, expected):
    tree = DecisionTreeRegressor(max_depth=5, min_size=2)
    result = tree._make_leaf(y)
    assert expected == pytest.approx(result)


get_params_tests = [
    ({"max_depth":3, "min_size":5, "n_features":None},
     {"max_depth":3, "min_size":5, "n_features":None})
]

@pytest.mark.parametrize('test_dict, expected_dict', get_params_tests)
def test_get_params(test_dict, expected_dict):
    tree = DecisionTreeRegressor(max_depth  = test_dict["max_depth"],
                                 min_size   = test_dict["min_size"],
                                 n_features = test_dict["n_features"])

    assert expected_dict == tree.get_params()
#
def test_default_getparams():
    tree = DecisionTreeRegressor()
    assert {"max_depth":2, "min_size":1, "n_features":None} == tree.get_params()


mse_cost_tests =[ ((np.array([[0.],[0.], [0.],[0.]]),
                    np.array([[0.],[0.], [1.],[1.]])),
                   0.25),
                   ((np.array([[0.],[1.], [0.],[0.]]),
                    np.array([[0.],[0.], [0.],[1.]])),
                   0.374999999)
                  ]

@pytest.mark.parametrize('groups, expected', mse_cost_tests)
def test_cost_mse(groups, expected):
    tree = DecisionTreeRegressor()
    result = tree._cost_mse(groups)
    assert expected == pytest.approx(result)



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
    assert tree.score(X,y) == pytest.approx(0.004999999)