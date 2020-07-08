import pytest
import numpy as np
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

@pytest.mark.parametrize('data, column, value, expected', test_split_data)
def test_split_dataset(data, column, value, expected):
  tree = DecisionTree(max_depth=5, min_size=2)
  result = tree._test_split(dataset = data, 
                            column  = column, 
                            value   = value) 

  assert( np.array_equal(result[0], expected[0]) & 
  	      np.array_equal(result[0], expected[0]))