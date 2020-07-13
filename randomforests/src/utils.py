import pandas as pd
import numpy as np

def _make_dataset(X, y):
	"""
	This function converts a Pandas Dataframe X and
	the Pandas Series y as a new numpy array


	Parameters
	---------
		X: The Pandas DataFrame of the dataset features

		y: The Pandas Series of the target values

	Returns
	--------
		np.ndarray The datafarme X with y appended to the right 
		most column 
	"""
	feats  = X
	target = y

	# convert the the dataframe/series to numpy array if 
	# not in numpy array format
	if isinstance(X, np.ndarray) is False:
		if isinstance(X, pd.core.frame.DataFrame) is True:
			feats =  X.to_numpy()
		else:
			raise TypeError("X needs to be NumPy array or Pandas Dataframe")

	if isinstance(y, np.ndarray) is False:
		if isinstance(y, pd.core.series.Series) is True:
			target = y.values
		else:
			raise TypeError("X needs to be NumPy array or Pandas Series")
	
	# append the column vector as the right most column
	dataset = np.append(feats,target.reshape(len(target),1),axis=1)

	return dataset