import pytest
import pandas as pd

#from TreeMethods.DecisionTree import TreeNode
from TreeMethods.ClassificationDecisionTree import DecisionTreeClassifier

def test_initialization():
	
	tree = DecisionTreeClassifier(max_depth=5, min_size=1)

	assert tree.max_depth == 5
	assert tree.min_size ==1 


test_gini_index_data = [
([[1,1,0], [1,1,0]], [[1,1,1], [1,1,1]], 0.0),
([[1,1,1], [1,1,0]], [[1,1,0], [1,1,1]], 1.0)
]

@pytest.mark.parametrize('df1, df2, expected', test_gini_index_data)
def test_gini_index(df1, df2, expected):
	
	tree = DecisionTreeClassifier(max_depth=5, min_size=1)
	tree.target_values = [0,1]
	assert tree._gini_index([df1, df2]) == expected



def test_get_split():
	tree = DecisionTreeClassifier(max_depth=5, min_size=2)

	df = [[1, 2, 3, 0],
		[1.2, 2, 3, 0],
		[1.3, 2, 3, 0],
		[2.0, 2, 3, 1],
		[2.1, 2, 3, 1],
		[2.5, 2, 3, 1]]

	tree.set_n_features(3)
	tree.columns = ['col1','col2','col3']
	tree.target_values = [0,1]

	result = tree._get_split(df)

	assert result['splitting_col'] == 0
	assert result['splitting_feature'] == 'col1'
	assert result['splitting_value'] == 2.0


test_make_leaf_data = [
([[3,0],[3,0],[3,0],[3,1],[3,1],[3,1],[3,1]], 1)
]

@pytest.mark.parametrize('df, expected', test_make_leaf_data)
def test_make_leaf(df, expected):

	tree = DecisionTreeClassifier(max_depth=5, min_size=2)

	tree.target_values = [0,1]

	assert tree._make_leaf(df) == expected

test_predict_data = [
(pd.Series([2.0, 23.0], index=['col1','col2']), 0),
(pd.Series([4.0, 23.0], index=['col1','col2']), 1),
(pd.Series([12.0, 23.0], index=['col1','col2']), 1)
]

@pytest.mark.parametrize('row, expected', test_predict_data)
def test_predict(row, expected):

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
	df.sort_values('col1',inplace=True)
	df = df.reset_index(drop=True)

	tree = DecisionTreeClassifier(2,1)
	tree.fit(df,target='tar')
	