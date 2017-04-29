import pytest
from TreeMethods.DecisionTree import DecisionTree
import pandas as pd

###############################################################################

test_split_datset_data =[
  ( [[1, 4, 3, 0],
	 [2, 3, 3, 0],
	 [3, 2, 3, 0],
	 [4, 1, 3, 1]], 
	 0, 3,
	 [[[1, 4, 3, 0],
	   [2, 3, 3, 0]],
	  [[3, 2, 3, 0],
	   [4, 1, 3, 1]]]
  ),
  ( [[1, 4, 3, 0],
	 [2, 3, 3, 0],
	 [3, 2, 3, 0],
	 [4, 1, 3, 1]], 
	 1, 2,
	 [[[4, 1, 3, 1]],
	  [[1, 4, 3, 0],
	   [2, 3, 3, 0],
	   [3, 2, 3, 0]]
	 ]
  )
  ]

@pytest.mark.parametrize('df, feature, value, expected', test_split_datset_data)
def test_split_dataset(df, feature, value, expected):
	tree = DecisionTree(max_depth=5, min_size=2)

	assert tree._test_split(feature,value,df) == tuple(expected)

###############################################################################

def test_convert_to_list():

	dataset = [[2.771244718,1.784783929,0],
	[1.728571309,1.169761413,0],
	[3.678319846,2.81281357,1],
	[3.961043357,2.61995032,1],
	[2.999208922,2.209014212,0],
	[7.497545867,3.162953546,0],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,1],
	[10.12493903,3.234550982,0],
	[6.642287351,3.319983761,1]]
	
	df = pd.DataFrame(data=dataset,columns =['col1','col2','tar'])

	tree = DecisionTree(2,1)

	dataset_2 = tree._convert_dataframe_to_list(df,'tar')

	assert dataset == dataset_2
	