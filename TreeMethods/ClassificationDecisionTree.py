import pandas as pd
import numpy as np

from DecisionTree import DecisionTree
from DecisionTree import TreeNode
import numpy as np

class DecisionTreeClassifier (DecisionTree):

	"""
	DecisionTree Classifier.
	"""

	
	def __init__(self, max_depth=2, min_size=5, error_function='gini'):
		"""
		Constructor for a classification decision tree.

		Kargs:
			max_depth (int) : The maximum depth of tree.
			min_size (int) : The minimum number of datapoints in terminal nodes.
			error_function (str) : The name of the error function.
		"""
		DecisionTree.__init__(self, max_depth, min_size)

		self.target_values = None
		self.error_function = error_function
	
	def fit(self, dataset, target):
		"""
		Builds the decsision tree by recursively splitting tree until the
		the maxmimum depth of the tree is acheived or the nodes have the
		min_size number of training points.
		
		Args:
			dataset (`DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_ ):
				Training data.
			
			target (str): The column name of the target in the dataset.
		"""
		self.original_n_features = dataset.shape[1] -1

		# If the number of features used to choose the split hasnt been specified
		# then set it the number of features in the dataset.
		if self.n_features == None:
			self.n_features = self.original_n_features

		# Make a new dataset that is list of lists
		new_dataset = self._convert_to_list(dataset,target)

		# get all the targe values
		self.target_values = list(set(row[-1] for row in new_dataset))

		# Get the first split of the dataset
		node_value = self._get_split(new_dataset)

		# Creates the root with val node_value
		self.root = TreeNode(node_value)

		# Now recursively split the tree
		self._split(self.root, new_dataset, 1)

	def predict(self, row):
		"""
		Predict the class that this datapoint belongs to.

		Args:
			row (`Series <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html>`_) : 
				The datapoint to classify.

		Returns:
			int. The class the data points belong to.
		"""
		return self._predict(row)

	def _error(self, groups):
		"""
		The error function which one wants to minimize on the split.
		will choose the proper error function.
	
		Args:
			groups (list) : list of the two subdatasets after splitting

		Returns:
			float. The error of the split
		"""
		return self._gini_index(groups)

	def _gini_index(self, groups):
		"""
		Returns the gini-index for the split.

		Args:
			groups (list) : list of the two subdatasets after splitting

		Returns:
			float. gini-index of the split
		"""
		gini = 0
		for target_value in self.target_values:
			for group in groups:
				size = len(group)
				if size == 0:
					continue
				p = float( [ row[-1] for row in group ].count(target_value)) / float(size)
				gini += p*(1-p)

		return gini


	def _make_leaf(self, group):
		"""
        Creates a terminal node value by selecting amoung the group that has
        the majority.

        Args: 
        	group (list): The dataset

       	Returns:
       		int. The majority of the class.
    	"""

    	# Get the target class values
		outcomes = [row[-1] for row in group]

		# Get the class value with the max number of counts.
		return max(set(outcomes), key=outcomes.count)
