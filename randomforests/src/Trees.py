
from random import randrange
import pandas as pd
import numpy as np
import math

from funcs import _make_dataset


class DecisionTree:
	"""
	A decision tree base class. 

	Classification and Regression Trees will be derived class that override 
	certain functions of this class.  This was done because many common
	methods, so to reduce code they are written here in the base class.
		
	:Atrributes:
		**max_depth** (int): The maximum depth of tree.

		**min_size** (int): The minimum number of datapoints in terminal nodes.

		**n_features** (int): The number of features to be used in splitting.

		**root** (dictionary): The root of the decision tree.

	"""
	def __init__(self, max_depth=2, min_size=1, n_features=None):
		"""
		Constructor for a generic decision tree.  It just sets the max_depth
		and min_size.

		Args: 
			max_depth (int) : The maximum depth of tree.
			min_size (int) : The min number of datapoints in terminal nodes.
		"""
		self.max_depth  = max_depth
		self.min_size   = min_size
		if n_features is not None:
			self.n_features = n_features-1
		else:
			self.n_features = None
		self.root       = None


	def _fit(self, X = None, Y = None):
		"""
		Builds the decsision tree by recursively splitting tree until the
		the maxmimum depth, max_depth, of the tree is acheived or the nodes
		have the minmum number of training points per node, min_size, is
		achieved.

		Note: n_features will be passed by the RandomForest as it is 
			  usually ta subset of the total number of features. 
			  However, if one is using the class as a stand alone decision
			  tree, then the n_features will automatically be 
		
		Args:
			train (list or DataFrame) : The dataset.

			target (str): The name of the target variable

			n_features (int) : The number of features.
		"""
        
		if self.n_features is None:
			self.n_features = len(dataset) - 1
            
		# perform optimal split for the root
		self.root = self._get_split(dataset)

		# now recurisively split the roots dataset until the stopping
		# criteria is met.
		root = self._split(self.root, 1)


	def _test_split(self, dataset, index, value):
		"""
		This function splits the data set depending on the feature (index) and
		the splitting value (value)

		Args:
			index (int) : The column index of the feature.
			value (float) : The value to split the data.
			dataset (list) : The list of list representation of the dataframe

		Returns:
			Tupple of the left and right split datasets.
		"""
		left  = dataset[dataset[:,index] < value]
		right = dataset[dataset[:,index] >= value]
		return left, right


	def _get_split(self, dataset):
		"""
		Select the best splitting point and feature for a dataset 
		using a random selection of self.n_features number of features.

		Args:	
			dataset (list of list): Training data.
			
		Returns:
			Dictionary of the best splitting feature of randomly chosen and 
			the best splitting value.
		"""
		b_index, b_value, b_score, b_groups = 999, 999, 999, None

		# the features to test among the split
		features = set()

		# randomily select features to consider
		while len(features) < self.n_features:
			index = randrange(len(dataset[0])-1)
			features.add(index)

		# loop through the number of features and values of the data
		# to figure out which gives the best split according
		# to the derived classes cost function value of the tested 
		# split
		for index in features:
			for row in dataset:
				groups = self._test_split(dataset, index, row[index])
				gini   = self._cost(groups)
				if gini < b_score:
					b_index  = index
					b_value  = row[index]
					b_score  = gini
					b_groups = groups

		return {'index':b_index, 'value':b_value, 'groups':b_groups}

	def _split(self, node, depth):
		"""
		Recursive splitting function that creates child
		splits for a node or make this node a leaf.
		Note: Leaves are just a value, which is determined
		in the derived class.

		Args:
			node (dictionary): The current node in the tree.

			depth (int) : The depth of node curr.

		Returns: None
		"""
		left, right = node['groups']
		del(node['groups'])
		# check for a no split
		if not left or not right:
			node['left'] = node['right'] = self._make_leaf(left + right)
			return
		# check for max depth
		if depth >= self.max_depth:
			node['left'] = self._make_leaf(left)
			node['right'] = self._make_leaf(right)
			return
		# process left child
		if len(left) <= self.min_size:
			node['left'] = self._make_leaf(left)
		else:
			node['left'] = self._get_split(left)
			self._split(node['left'], depth+1)
		# process right child
		if len(right) <= self.min_size:
			node['right'] = self._make_leaf(right)
		else:
			node['right'] = self._get_split(right)
			self._split(node['right'], depth+1)

	def _predict(self, row, node):
		"""
		Predicts the target value that this datapoint belongs to by recursively
		traversing tree and returns the termina leaf value corresponding 
		to this data point.

		Args:
			row (list ) : The data point to classify.

			node (dict ) : The current node in the tree.

		Returns:
			The leaf value of this data point.
		"""
		if row[node['index']] < node['value']:
			if isinstance(node['left'], dict):
				return self._predict(row, node['left'])
			else:
				return node['left']
		else:
			if isinstance(node['right'], dict):
				return self._predict(row, node['right'])
			else:
				return node['right']

	
	def print_tree(self, node=None, depth=0):
		"""
		Prints the tree using a pre-order traversal.

		:Parameters: 
		
			**node** (dict) : Current node in the tree.

			**depth** (int) : The depth of the current node.
		"""
		if node is None:
			self.print_tree(self.root)
		else:
			if isinstance(node, dict):
				print('%s[X%d < %.3f]' % ((depth*' ', 
					  (node['index']+1), node['value'])))
				self.print_tree(node['left'], depth+1)
				self.print_tree(node['right'], depth+1)
			else:
				print('%s[%s]' % ((depth*' ', node)))


class DecisionTreeClassifier (DecisionTree):
	"""
	A decision tree classifier that extends the DecisionTree class. 

	:Attributes: 
		**max_depth** (int): The maximum depth of tree.

		**min_size** (int): The minimum number of datapoints in terminal nodes.

		**n_features** (int): The number of features to be used in splitting.

		**root** (dictionary): The root of the decision tree.

		**columns** (list) : The feature names.

		**class_values** (list) : The list of the target class values.

		**cost_function** (str) : The name of the cost function to use: 'gini'.
	"""

	def __init__(self, max_depth=2, min_size=2, cost='gini'):
		"""
		Constructor for the Decision Tree Classifer.  It calls the base
		class constructor and sets the cost function.  If the cost
		parameter is not 'gini' then an exception is thrown.

		Parameters: 
			**max_depth** (int): The maximum depth of tree.

			**min_size** (int): The minimum number of datapoints in terminal nodes.

			**cost** (str) : The name of the cost function to use: 'gini'.
		"""
		DecisionTree.__init__(self, max_depth, min_size)
		self.class_values  = None
		self.cost_function = None
		if cost == 'gini':
			self.cost_function = cost
		else:
			raise NameError('Not valid cost function')

		

	def fit(self, X=None, Y=None):
		"""
		Builds the classification decsision tree by recursively splitting 
		tree until the the maxmimum depth, max_depth of the tree is acheived or
		the node have the minimum number of training points, min_size.
		
		n_features will be passed by the RandomForest as it is usually a subset 
		of the total number of features. However, if one is using the class as a 
		stand alone decision tree, then the n_features will automatically be 
		
		:Parameters:
			**dataset** (list or `Pandas DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_) : The dataset.
	
			**target** (str) : The name of the target variable.

			**n_features*** (int) : The number of features.
		"""
		self.class_values = list(set(row[-1] for row in train))
		self._fit(train, target)

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
			float. Either the gini-index or entropy of the split.

		"""
		return self._gini_index(groups)

	def _gini_index(self, groups):
		"""
		Returns the gini-index for the split of the dataset into two groups.

		Args:
			groups (list) : List of the two subdatasets after splitting.

		Returns:
			float. gini-index of the split.
		"""
		gini = 0.0
		for class_value in self.class_values:
			for group in groups:
				size = len(group)
				if size == 0:
					continue
				p = [row[-1] for row in group].count(class_value) / float(size)
				gini += (p * (1.0 - p))
		return gini


	def _make_leaf(self, group):
		"""
        Creates a terminal node value by selecting amoung the group that has
        the majority.

        Args: 
        	group (list): The subgroup of the dataset.

       	Returns:
       		int. The majority of this groups target class values.
    	"""
		outcomes = group[:,-1] #[row[-1] for row in group]
		return max(set(outcomes), key=outcomes.count)



