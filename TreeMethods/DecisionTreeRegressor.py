import numpy as np
from DecisionTree import DecisionTree


class DecisionTreeRegressor (DecisionTree):
	"""
	A decision tree regression model that extends the DecisionTree class. 

	:Attributes: 
		**max_depth** (int): The maximum depth of tree.

		**min_size** (int): The minimum number of datapoints in terminal nodes.

		**n_features** (int): The number of features to be used in splitting.

		**root** (dictionary): The root of the decision tree.

		**columns** (list) : The feature names.

		**cost_function** (str) : The name of the cost function to use: 'mse'
	"""

	def __init__(self, max_depth=2, min_size=2, cost='mse'):
		"""
		Constructor for the Decision Tree Regressor.  It calls the base
		class constructor and sets the cost function.  If the cost
		parameter is not 'mse' an error is thrown.

		Parameters: 
			**max_depth** (int): The maximum depth of tree.

			**min_size** (int): The minimum number of datapoints in terminal nodes.

			**cost** (str) : The name of the cost function to use: 'gini' or 'entropy'
		"""
		DecisionTree.__init__(self, max_depth, min_size)
		self.cost_function = None
		if cost == 'mse':
			self.cost_function = cost
		else:
			raise NameError('Not valid cost function')


	def fit(self, train, target=None, n_features=None):
		"""
		Builds the regression decsision tree by recursively splitting 
		tree until the the maxmimum depth, max_depth of the tree is acheived or
		the node have the minimum number of training points, min_size.
		
		n_features will be passed by the RandomForest as it is usually a subset 
		of the total number of features. However, if one is using the class as 
		a stand alone decision tree, then the n_features will automatically be 
		
		:Parameters:
			**dataset** (list or `Pandas DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_) : The dataset.
	
			**target** (str) : The name of the target variable.

			**n_features** (int) : The number of features.
		"""
		self._fit(train, target, n_features)

	def predict(self, row):
		"""
		Predict the class that this sample datapoint belongs to.

		:Parameter: **row** (list or `Pandas Series <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html>`_) : The datapoint to classify.

		:Returns: (int) The class the data points belong to.
		"""
		if isinstance(row, list) is False:
			return self._predict(row.tolist(), self.root)
		else:
			return self._predict(row, self.root)

	def _cost(self, groups):
		"""
		Returns the associated cost for the split of the dataset 
		into two groups. The cost_function will be set when the
		tree is initialized.

		Args: 
			groups (list) : List of the two subdatasets after splitting.

		Returns:
			float. Mean square error of the split.

		"""
		return self._mse(groups)

	def _mse(self, groups):
		"""
		Returns the mse for the split of the dataset into two groups.

		Args:
			groups (list) : List of the two subdatasets after splitting.

		Returns:
			float. mse of the split.
		"""
		mse = 0.0
		for group in groups:
			if len(group) == 0:
					continue
			outcomes = [row[-1] for row in group]
			mse += np.std(outcomes)
		return mse

	def _make_leaf(self, group):
		"""
        Creates a terminal node value by taking the average target value
		in the group.

        Args: 
        	group (list): The subgroup of the dataset.

       	Returns:
       		int. The mean  of this groups target class values.
    	"""
		outcomes = [row[-1] for row in group]
		return np.mean(outcomes)