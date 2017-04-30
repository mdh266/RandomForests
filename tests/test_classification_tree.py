import pytest
import pandas as pd

#from TreeMethods.DecisionTree import TreeNode
from TreeMethods.DecisionTreeClassifier import DecisionTreeClassifier

def test_initialization():
	
	tree = DecisionTreeClassifier(max_depth=5, min_size=1)

	assert tree.max_depth == 5
	assert tree.min_size ==1 

###############################################################################

test_gini_index_data = [
(([[1,1,0], [1,1,0]], [[1,1,1], [1,1,1]]), 0.0),
(([[1,1,1], [1,1,0]], [[1,1,0], [1,1,1]]), 1.0)
]

@pytest.mark.parametrize('df, expected', test_gini_index_data)
def test_gini_index(df, expected):
	
	tree = DecisionTreeClassifier(max_depth=5, min_size=1)
	tree.class_values = [0,1]
	assert tree._gini_index(df) == expected

###############################################################################

test_make_leaf_data = [
([[3,0],[3,0],[3,0],[3,1],[3,1],[3,1],[3,1]], 1)
]

@pytest.mark.parametrize('df, expected', test_make_leaf_data)
def test_make_leaf(df, expected):

	tree = DecisionTreeClassifier(max_depth=5, min_size=1)
	tree.class_values = [0,1]

	assert tree._make_leaf(df) == expected

###############################################################################


def test_get_split():
	tree = DecisionTreeClassifier(max_depth=5, min_size=2)

	df = [[1, 2, 3, 0],
		[1.2, 2, 3, 0],
		[1.3, 2, 3, 0],
		[2.0, 2, 3, 1],
		[2.1, 2, 3, 1],
		[2.5, 2, 3, 1]]

	tree.n_features = 3
	tree.columns = ['col1','col2','col3']
	tree.class_values = [0,1]

	result = tree._get_split(df)

	assert result['index'] == 0
	assert result['value'] == 2.0

###############################################################################
test_predict_data = [
(pd.Series([1.0], index=['col1']), 0),
(pd.Series([4.0], index=['col1']), 1)]

#@pytest.mark.parametrize('row, expected', test_predict_data)
#def test_predict(row, expected):
#	tree = DecisionTreeClassifier(max_depth=5, min_size=2)

#	dataset = [[1, 0],
#		[1.2, 0],
#		[1.3, 0],
#		[2.0, 1],
#		[2.1, 1],
#		[2.5, 1]]

	
#	df = pd.DataFrame(data=dataset,columns =['col1','tar'])
	#df.sort_values('col1',inplace=True)
#	df = df.reset_index(drop=True)

#	tree = DecisionTreeClassifier(5,2) 
#	tree.fit(df,target='tar')
	
#	assert tree.predict(row) == expected


