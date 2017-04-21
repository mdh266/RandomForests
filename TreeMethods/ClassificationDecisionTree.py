import pandas as pd
import numpy as np

from DecisionTree import DecisionTree
from DecisionTree import TreeNode
import numpy as np

class DecisionTreeClassifier (DecisionTree):
	
	def __init__(self, max_depth=2, min_size=5):
		"""
		Constructor for a classification decision tree.

		:param int max_depth: The maximum depth of tree.
		:param int min_size: The minimum number of datapoints in terminal nodes.
		"""
		DecisionTree.__init__(self, max_depth, min_size)

		self.columns = None
		self.target_values = None
	
	def fit(self, dataset, target):
		"""
		Builds the decsision tree by recursively splitting tree until the
		the maxmimum depth of the tree is acheived or the nodes have the
		min_size number of training points.
		
		:parameters:
			**dataset** (`DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_ ):
				Training data.
			
			**target** (str): 
				The column name of the target in the dataset.
		"""
		self.original_n_features = dataset.shape[1] -1

		# If the number of features used to choose the split hasnt been specified
		# then set it the number of features in the dataset.
		if self.n_features == None:
			self.n_features = self.original_n_features

		# set the column names 
		self.columns = dataset.columns

		# Make a new dataset that is list of lists
		new_dataset = self._convert_to_list(dataset,target)

		# get all the targe values
		self.target_values = list(set(row[-1] for row in dataset))

		# Get the first split of the dataset
		node_value = self._get_split(new_dataset)

		# Creates the root with val node_value
		self.root = TreeNode(node_value)


		# Now recursively split the tree
		self._split(self.root, dataset, 1)

	def _error(self, groups):
		return self._gini_index(groups)

	def _gini_index(self, groups):
		"""
		Returns the gini-index for the split.

		:param: groups (list) : list of the two subdatasets after splitting
		:return: gini-index of the split
		:rtype: float
		"""
		gini = 0
		for target_value in self.target_values:
			for group in groups:
				size = len(group)
				if size == 0:
					continue
				p = float( [ row[-1] for row in group ].count(target_value)) 
				p /= float(size)
				gini += p*(1-p)

		return gini


	def _make_leaf(self, group):
		"""
        Creates a terminal node value by selecting amoung the group that has
        the majority.

        :param: group (list(list)): The dataset

        :return: The majority of the class.
        :rtype: int
    	"""

		# not sure i need this check
		outcomes = [row[-1] for row in group]
		return max(set(outcomes), key=outcomes.count)
		

		return value