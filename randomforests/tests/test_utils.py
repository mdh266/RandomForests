import pytest
import pandas as pd 
import numpy as np

from src.utils import _make_dataset 

def test_make_datatset():

	dataset = np.array([[2.771244718,1.784783929,0],
					[1.728571309,1.169761413,0],
					[3.678319846,2.81281357,1],
					[3.961043357,2.61995032,1],
					[2.999208922,2.209014212,0],
					[7.497545867,3.162953546,0],
					[9.00220326,3.339047188,1],
					[7.444542326,0.476683375,1],
					[10.12493903,3.234550982,0],
					[6.642287351,3.319983761,1]])


	df = pd.DataFrame(data=dataset,columns =['col1','col2','tar'])
	X  = df[["col1","col2"]]
	y  = df["tar"]

	assert np.array_equal(dataset, _make_dataset(X=X, y=y))


dataset_test = [(1,2),(None,None)]


@pytest.mark.parametrize('X, y', dataset_test)
def test_make_dataset_errors(X,y):
	with pytest.raises(Exception):
		_make_dataset(X,y)