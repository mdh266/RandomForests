from math import sqrt
from RandomForest import RandomForest
from DecisionTreeClassifier import DecisionTreeClassifier



class RandomForestClassifier (RandomForest):

	"""
	A random forest classifier that derives from the base class RandomForest.
		
	:Attributes: 
		**n_trees** (int) : The number of trees to use.

		**max_depth** (int): The maximum depth of tree.

		**min_size** (int): The minimum number of datapoints in terminal nodes.

		**cost_function** (str) : The name of the cost function to use: 'gini'.

		**trees** (list) : A list of the DecisionTree objects.

		**columns** (list) : The feature names.
	"""

	def __init__(self, n_trees=10, max_depth=2, min_size=2, cost='gini'):
		"""
		Constructor for random forest classifier. This mainly just initializez
		the attributes of the class by calling the base class constructor. 
		However, here is where it is the cost function string is checked
		to make sure it either using 'gini', otherwise an error is thrown.

		Args:
			cost (str) : The name of the cost function to use for evaluating
						 the split.

			n_trees (int): The number of trees to use.

			max_depth (int): The maximum depth of tree.

			min_size (int): The minimum number of datapoints in terminal nodes.
	
		"""
		if cost != 'gini':
			raise NameError('Not valid cost function')
		else:
			RandomForest.__init__(self, cost,  n_trees=10, max_depth=2, min_size=2)
		

	def fit(self, train, target=None, test=None):
		"""
		Fit the random forest to the training set train.  If a test set is provided
		then the return value wil be the predictions of the RandomForest on the
		test set.  If no test set is provide nothing is returned.


		Note: Below we set the number of features to use in the splitting to be
		the square root of the number of total features in the dataset.

		:Parameters:
			**train** (list or `Pandas DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_) : The training set.
			
			**target** (str or None) : The name of the target variable
			
			**test** (list or `Pandas DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_) : The test set.
			
		:Returns:
			(list or None): If a test set is provided then the return value wil be
			the predictions of the RandomForest on the test set.  If no test set 
			is provide nothing is returned.
		"""
		# set the number of features for the trees to use.
		if isinstance(train, list) is False:
			if target is None:
				raise ValueError('If passing dataframe need to specify target.')
			else:
		
				train = self._convert_dataframe_to_list(train, target)
	
		n_features = int(sqrt(len(train[0])-1))

		for i in range(self.n_trees):
			sample = self._subsample(train)
			tree = DecisionTreeClassifier(self.max_depth,
										   self.min_size,
										   self.cost_function)

			tree.fit(sample, n_features)
			self.trees.append(tree)

		# if the test set is not empty then return the predictions
		if test is not None:
			predictions = [self.predict(row) for row in test]
			return(predictions)


	def predict(self, row):
		"""
		Peform a prediction for a sample data point by bagging 
		the prediction of the trees in the ensemble. The majority
		target class that is chosen.

		:Parameter: **row** (list or `Pandas Series <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html>`_ ) : The data point to classify.

		:Returns: (int) : The predicted target class for this data point.
		"""
		if isinstance(row, list) is False:
			row = row.tolist()
			predictions = [tree.predict(row) for tree in self.trees]
		else:
			predictions = [tree.predict(row) for tree in self.trees]

		return max(set(predictions), key=predictions.count)


	def KFoldCV(self, dataset, n_folds=10):
		"""
		Perform k-fold cross validatation on the dataset 
		and return the acrruracy of each training.

		:Parameters:
			**dataset** (list or `Pandas DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_) : The dataset in list form.
			
			**n_fold** (int) : The number of folds in the k-fold CV.

		:Returns: (list) : List of the accuracy of each Random Forest on each
			of the folds.
		"""
		if isinstance(dataset, list) is False:
			if target is None:
				raise ValueError('If passing dataframe need to specify target.')
			else:
				dataset = self._convert_dataframe_to_list(dataset, target)

		folds = self._cross_validation_split(dataset, n_folds)
		scores = list()
		for fold in folds:
			train_set = list(folds)
			train_set.remove(fold)
			train_set = sum(train_set, [])
			test_set = list()
			for row in fold:
				row_copy = list(row)
				test_set.append(row_copy)
				row_copy[-1] = None
			predicted = self.fit(train_set, test_set)
			actual = [row[-1] for row in fold]
			accuracy = self._metric(actual, predicted)
			scores.append(accuracy)
		return scores


	def _metric(self, actual, predicted):
		"""
		Returns the accuracy of the predictions for now, extending it 
		to include other metrics.

		Args: 
			actual (list) : The actual target values.
			predicted (list) : The predicted target values.

		Returns:
			float.  The accuracy of the predictions.

		"""
		return self._accuracy(actual, predicted)

	def _accuracy(self, actual, predicted):
		"""
		Computes the accuracy by counting how many predictions were correct.

		Args: 
			actual (list) : The actual target values.
			predicted (list) : The predicted target values.

		Returns:
			float.  The accuracy of the predictions.
		"""
		correct = 0
		for i in range(len(actual)):
			if actual[i] == predicted[i]:
				correct += 1
		return correct / float(len(actual)) * 100.0


