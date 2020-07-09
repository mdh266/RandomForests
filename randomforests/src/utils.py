import pandas as pd
import numpy as np

def _make_dataset(X, y):
	"""
	This function converts a Pandas Dataframe 

	Parameters
	---------
		X: The Pandas DataFrame of the dataset features

		y: The Pandas Series of the target values

	Returns
	--------
		List representation of the datafarme with y
		appended to the right most column

	"""
	feats = X
	feats = y

	# convert the the dataframe/series to numpy array if 
	# not in numpy array format
	if isinstance(X, np.ndarray) is False:
		if isinstance(X, pd.core.frame.DataFrame) is True:
			feats =  X.to_numpy()

	if isinstance(y, np.ndarray) is False:
		if isinstance(y, pd.core.series.Series) is True:
			target = y.values
	
	# append the column vector
	dataset = np.append(feats,target.reshape(len(target),1),axis=1)

	return dataset