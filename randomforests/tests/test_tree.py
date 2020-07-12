import pytest
import numpy as np
import pandas as pd
from src.Trees import DecisionTree


test_split_data =[
  ( np.array([[1, 4, 3, 0],
              [2, 3, 3, 0],
              [3, 2, 3, 0],
              [4, 1, 3, 1]]), 
   0, 3,
   [np.array([[1, 4, 3, 0],
              [2, 3, 3, 0]]),
    np.array([[3, 2, 3, 0],
               [4, 1, 3, 1]])
  ]),
  ( np.array([[1, 4, 3, 0],
              [2, 3, 3, 0],
              [3, 2, 3, 0],
              [4, 1, 3, 1]]), 
   1, 2,
   [np.array([[4, 1, 3, 1]]),
    np.array([[1, 4, 3, 0],
              [2, 3, 3, 0],
              [3, 2, 3, 0]])
   ])
  ]

predict_tests = [(np.array([1,2]), {"column":0, "value":2, "left":1}, 1),
                 (np.array([1,2]), {"column":0, "value":1, "right":0}, 0),
                 (np.array([1,2]), {"column":1, "value":3, "left":1}, 1),
                 (np.array([1,2]), 
                  {"column":0, "value":1, "right": 
                  {"column":1, "value":5, "left": 0}}, 0)
                ]


@pytest.mark.parametrize('data, column, value, expected', test_split_data)
def test_split_dataset(data, column, value, expected):
  tree = DecisionTree(max_depth=5, min_size=2)
  result = tree._test_split(dataset = data, 
                            column  = column, 
                            value   = value) 

  assert( np.array_equal(result[0], expected[0]) & 
  	      np.array_equal(result[0], expected[0]))


@pytest.mark.parametrize('row, node, expected', predict_tests)
def test__predict(row, node, expected):
  tree = DecisionTree(max_depth=5, min_size=2)
  result = tree._predict(row = row, node = node)
  assert expected == result


def test_init():

  tree = DecisionTree(3,2,1)

  assert (tree.max_depth == 3 and
          tree.min_size  == 2 and
          tree.n_features == 1)

def test_set_features():
  tree = DecisionTree()
  X    = pd.DataFrame({"x1":[0,1],"x2":[1,0]})
  tree._set_features(X)

  assert tree.n_features == 2

def test_set_features_error():
  tree = DecisionTree(n_features=5)
  X    = pd.DataFrame({"x1":[0,1],"x2":[1,0]})
  with pytest.raises(Exception):
    tree._set_features(X)

