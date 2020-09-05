import pytest
import numpy as np
from randomforests import DecisionTreeRegressor


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


def test_default_getparams():
    tree = DecisionTreeRegressor()
    assert {"max_depth":2, "min_size":1, "n_features":None} == tree.get_params()



mse_cost_tests =[ ((np.array([[0.],[0.], [0.],[0.]]),
                    np.array([[0.],[0.], [1.],[1.]])),
                   0.125),
                   ((np.array([[0.],[1.], [0.],[0.]]),
                    np.array([[0.],[0.], [0.],[1.]])),
                   0.1874999995)
                  ]

@pytest.mark.parametrize('groups, expected', mse_cost_tests)
def test_cost_mse(groups, expected):
    tree = DecisionTreeRegressor()
    result = tree._cost_mse(groups)
    assert expected == pytest.approx(result)


