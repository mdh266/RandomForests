import pytest
import pandas as pd

from TreeMethods.DecisionTree import TreeNode
from TreeMethods.DecisionTree import DecisionTree

def test_test_split():

	tree = DecisionTree(
						max_depth=5,
						min_size=1)

	left = [[1, 2, 3, 0],
        	[1.2, 2, 3, 0],
        	[1.3, 2, 3, 0]]

	right = [[2.0, 2, 3, 1],
        	[2.1, 2, 3, 1],
        	[2.5, 2, 3, 1]]

	df1 = pd.DataFrame(data=left, columns=['col1','col2','col3','tar'])
	df2 = pd.DataFrame(data=right, columns=['col1','col2','col3', 'tar'])

	df = pd.concat([df1,df2])

	left_result, right_result = tree._test_split('col1', 1.5, df)
	
	assert df1.equals(left_result)
	assert df2.equals(right_result)



