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
                     0.5),
                   ((np.array([[0.],[1.], [0.],[0.]]),
                    np.array([[0.],[0.], [0.],[1.]])),
                     0.75)
]

@pytest.mark.parametrize('groups, expected', gini_cost_tests)
def test_cost_gini(groups, expected):
  tree = DecisionTreeClassifier()
  result = tree._cost_gini(groups)
  assert expected == result


get_split_tests = [
		(1,
		np.array([[0.1,0],[0.5,0],[0.7,1],[0.9,1]]),
		{'column': 0, 'value': 0.7, 'groups': 
		(np.array([[0.1, 0. ],[0.5, 0. ]]), 
		 np.array([[0.7, 1. ],[0.9, 1. ]]))}
		),

		(2,
    np.array([[0,0.1,0],[0,0.5,0],[0,0.7,1],[0,0.9,1]]),
    {'column': 1, 'value': 0.7, 'groups': 
    (np.array([[0. , 0.1, 0. ],[0. , 0.5, 0. ]]), 
     np.array([[0. , 0.7, 1. ],[0. , 0.9, 1. ]]))}
		)]

@pytest.mark.parametrize('n_features, dataset, expected', get_split_tests)
def test_get_split(n_features, dataset, expected):
  tree     = DecisionTreeClassifier(n_features = n_features)
  result   = tree._get_split(dataset)
  column   = result["column"] == expected["column"]
  value    = result["value"] == expected["value"]

  left_grp = np.array_equal(result["groups"][0],expected["groups"][0])
  right_grp= np.array_equal(result["groups"][1],expected["groups"][1])

  assert (column and value and left_grp and right_grp)



split_tests = [
      ({'column': 0,
       'value' : 0.0,
       'groups': (np.array([]), np.array([[0. , 0.1, 0 ], [0. , 0.5, 0]]))},
       1, # depth
       {'column': 0, 'value': 0.0, 'left': 0, 'right': 0}),
      ({'column': 0,
       'value' : 0.0,
       'groups': (np.array([]), np.array([[0. , 0.1, 0 ], [0. , 0.5, 0]]))},
       1, # depth
       {'column': 0, 'value': 0.0, 'left': 0, 'right': 0}),
      ({'column': 1,
       'value' : 9.0,
       'groups': (np.array([[0,1,1], [0. , 0.5, 1 ]]), 
                  np.array([[0,1,0], [0. , 0.5, 0 ]]))},
       2, # depth
       {'column': 1, 'value': 9.0, 'left': 1, 'right': 0})
]

@pytest.mark.parametrize('test_node, depth, expected', split_tests)
def test__split(test_node, depth, expected):
  tree   = DecisionTreeClassifier(max_depth = 2, min_size= 1, n_features = 2)
  tree._split(test_node, depth)

  assert test_node == expected



# @pytest.mark.xfail(raises=AttributeError)
# def test_get_split_exception():
#   tree = DecisionTreeClassifier(n_features=2)
#   tree._get_split(np.array([[0,1],[1,0]]))
