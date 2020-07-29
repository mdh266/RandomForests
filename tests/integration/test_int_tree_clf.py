import pytest
import numpy as np
import pandas as pd
from randomforests.TreeClassifier import DecisionTreeClassifier


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
    tree      = DecisionTreeClassifier(n_features = n_features)
    result    = tree._get_split(dataset)
    column    = result["column"] == expected["column"]
    value     = result["value"]  == expected["value"]

    left_grp  = np.array_equal(result["groups"][0],expected["groups"][0])
    right_grp = np.array_equal(result["groups"][1],expected["groups"][1])

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
     {'column': 1, 'value': 9.0, 'left': 1, 'right': 0}),
    ({'column': 1,
      'value': 0.7,
      'groups': (np.array([[0. , 0.1, 0 ]]),
                 np.array([[0. , 0.7, 1], [0. , 0.8, 0 ],[0. , 0.9, 1]]))},
     2, # depth
     {'column': 1, 'value': 0.7, 'left': 0, 'right': 1}),
    ({'column': 1,
      'value': 0.7,
      'groups': (np.array([[0. , 0.7, 1], [0. , 0.8, 0 ],[0. , 0.9, 1]]),
                 np.array([[0. , 0.1, 0 ]]))},
     2, # depth
     {'column': 1, 'value': 0.7, 'left': 1, 'right': 0})
]

@pytest.mark.parametrize('test_node, depth, expected', split_tests)
def test__split(test_node, depth, expected):
    tree   = DecisionTreeClassifier(max_depth = 2, min_size= 2, n_features = 2)
    tree._split(test_node, depth)

    assert test_node == expected


public_predict_test = [
    (np.array([[0. , 0.1],
               [0. , 0.5],
               [0. , 0.7],
               [0. , 0.9]]),
     np.array([0, 0, 1, 1])), # y
    (pd.DataFrame({"c1":[0., 0., 0., 0.],
                   "c2":[0.1, 0.5, 0.7, 0.9]}),
     pd.Series([0, 0, 1, 1]))
]

@pytest.mark.parametrize('X, y', public_predict_test)
def test_predict(X, y):
    tree   = DecisionTreeClassifier()
    model  = tree.fit(X, y)
    assert np.array_equal(model.predict(X),y)
