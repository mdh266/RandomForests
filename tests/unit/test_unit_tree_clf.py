import pytest
import numpy as np
from randomforests.TreeClassifier import DecisionTreeClassifier


leaf_tests =[(np.array([[0.],[0.], [1.],[0.]]),0),
             (np.array([[0.],[1.], [1.],[1.]]),1),
             (np.array([[1.],[1.], [1.],[1.]]),1)]



@pytest.mark.parametrize('y, expected', leaf_tests)
def test_make_leaf(y, expected):
  tree = DecisionTreeClassifier(max_depth=5, min_size=2)
  result = tree._make_leaf(y)
  assert expected == pytest.approx(result)


gini_index_tests =[
              (np.array([[0.],[0.], [0.],[0.]]),0),
	            (np.array([[0.],[0.], [1.],[1.]]),0.5),
	            (np.array([[0.],[0.], [0.],[1.]]),0.375),
	            (np.array([[1.],[1.], [0.],[1.]]),0.375),
	            (np.array([[1.],[1.], [0.]]),     0.4444444),
	            (np.array([[1.],[1.], [1.],[1.]]),0)]

@pytest.mark.parametrize('y, expected', gini_index_tests)
def test_gini_index(y, expected):
  tree = DecisionTreeClassifier(max_depth=5, min_size=2)
  result = tree._gini_index(y)
  assert expected == pytest.approx(result)


gini_cost_tests =[ ((np.array([[0.],[0.], [0.],[0.]]),
                     np.array([[0.],[0.], [1.],[1.]])),
                     0.25),
                   ((np.array([[0.],[1.], [0.],[0.]]),
                    np.array([[0.],[0.], [0.],[1.]])),
                     0.375)
]

@pytest.mark.parametrize('groups, expected', gini_cost_tests)
def test_cost_gini(groups, expected):
  tree = DecisionTreeClassifier()
  result = tree._cost_gini(groups)
  assert expected == result



get_params_tests = [
    ({"max_depth":3, "min_size":5, "n_features":None, "cost":'gini'},
     {"max_depth":3, "min_size":5, "n_features":None, "cost":'gini'})
]

@pytest.mark.parametrize('test_dict, expected_dict', get_params_tests)
def test_get_params(test_dict, expected_dict):
    tree = DecisionTreeClassifier(max_depth  = test_dict["max_depth"],
                                  min_size   = test_dict["min_size"],
                                  n_features = test_dict["n_features"])

    assert expected_dict == tree.get_params()

def test_default_getparams():
    tree = DecisionTreeClassifier()
    assert {"max_depth":2, "min_size":1, "n_features":None, "cost":'gini'} == tree.get_params()


def test_cost_exception():
    with pytest.raises(Exception):
        tree = DecisionTreeClassifier(cost="mse")


