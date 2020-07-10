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
  index    = result["index"] == expected["index"]
  value    = result["value"] == expected["value"]

  left_grp = np.array_equal(result["groups"][0],expected["groups"][0])
  right_grp= np.array_equal(result["groups"][1],expected["groups"][1])

  assert (index and value and left_grp and right_grp)



# @pytest.mark.xfail(raises=AttributeError)
# def test_get_split_exception():
#   tree = DecisionTreeClassifier(n_features=2)
#   tree._get_split(np.array([[0,1],[1,0]]))
