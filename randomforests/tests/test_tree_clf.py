import pytest

import numpy as np
from src.Trees import DecisionTreeClassifier


leaf_tests =[(np.array([[0.],[0.], [1.],[0.]]),0),
             (np.array([[0.],[1.], [1.],[1.]]),1),
             (np.array([[1.],[1.], [1.],[1.]]),1)]



@pytest.mark.parametrize('y, expected', leaf_tests)
def test_make_leaf(y, expected):
  tree = DecisionTreeClassifier(max_depth=5, min_size=2)
  result = tree._make_leaf(y)
  assert expected == pytest.approx(result)


gini_tests =[(np.array([[0.],[0.], [0.],[0.]]),0),
	            (np.array([[0.],[0.], [1.],[1.]]),0.5),
	            (np.array([[0.],[0.], [0.],[1.]]),0.375),
	            (np.array([[1.],[1.], [0.],[1.]]),0.375),
	            (np.array([[1.],[1.], [0.]]),     0.4444444),
	            (np.array([[1.],[1.], [1.],[1.]]),0)]

@pytest.mark.parametrize('y, expected', gini_tests)
def test_gini_index(y, expected):
  tree = DecisionTreeClassifier(max_depth=5, min_size=2)
  result = tree._gini_index(y)
  assert expected == pytest.approx(result)


get_split_tests = [
		(1,
		np.array([[0.1,0],[0.5,0],[0.7,1],[0.9,1]]),
		{'index': 0, 'value': 0.7, 'groups': 
		(np.array([[0.1, 0. ],[0.5, 0. ]]), 
		 np.array([[0.7, 1. ],[0.9, 1. ]]))}
		),

		(2,
    np.array([[0,0.1,0],[0,0.5,0],[0,0.7,1],[0,0.9,1]]),
    {'index': 1, 'value': 0.7, 'groups': 
    (np.array([[0. , 0.1, 0. ],[0. , 0.5, 0. ]]), 
     np.array([[0. , 0.7, 1. ],[0. , 0.9, 1. ]]))}
		)]

@pytest.mark.parametrize('n_features, dataset, expected', get_split_tests)
def test_get_split(n_features, dataset, expected):
  tree     = DecisionTreeClassifier(n_features = n_features)
  result   = tree._get_split(dataset)
  is_same  = result["index"] == expected["index"]
  is_same &= result["value"] == expected["value"]

  for res_grp, exp_grp in zip(result["groups"],expected["groups"]):
    is_same &= np.array_equal(res_grp, exp_grp)

  assert is_same



# @pytest.mark.xfail(raises=AttributeError)
# def test_get_split_exception():
#   tree = DecisionTreeClassifier(n_features=2)
#   tree._get_split(np.array([[0,1],[1,0]]))
