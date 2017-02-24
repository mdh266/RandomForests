from random import randrange  
import pandas as pd

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

		# Get the first split of the dataset
		node_value = self._get_split(dataset, 
							  		target)

		# Creates the root with val node_value
		self.root = TreeNode(node_value)

		# Now recursively split the tree
		self._split(self.root,
					dataset,
					target,
					1)


	def _get_split(self, dataset, target):
		"""
		Select the best split point and feature for a dataset using a random Selection of the features.

		:parameters:

 			**dataset** (`DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_ ):
				Training data.
			
			**target** (str): 
				The column name of the target in the dataset.

		:return: Dictionary of the best splitting feature of randomly chosen and the best splitting value.
		:rtype: dict
        """
		best_feature, best_value, best_score, best_groups = 999,999,999,None

		# the features to test among the split
		features = list()

		# this is for the testing 
		if self.columns is None:
			self.columns = dataset.columns

		# randomily select features to consider 
		while len(features) < self.n_features:

			# out of all the features
			feature_col = randrange(dataset.shape[1]-1)
 
			# exclude the target as a possible feature col
			if self.columns[feature_col] not in features and self.columns[feature_col] != target:
				features.append(self.columns[feature_col])
		
		#print features
		# loop through the number of features to figure out which
		# gives the best split.
		for feature in features:
			# split the data set according to this feature
			# and find the splits gini_index
			#print "feature : " + feature
			split_val_and_gini_index = self._find_best_split_value(dataset, feature, target)

			gini = split_val_and_gini_index[1]

			# if this is the best split update the info
			if gini < best_score:
				best_feature = feature
				best_value = split_val_and_gini_index[0]
				best_score = gini
					#best_groups = [left_group, right_group]

		return {'splitting_feature': best_feature,
                'splitting_value': best_value}

	def _find_best_split_value(self, dataset, feature, target):
		"""
		Select the best split point in the data set for the specificed features.

		:parameters:

 			**dataset** (`DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_ ):
				Training data.
		
			**feature** (str): 
				The column name of the feature to perform the search for best split on.s

			**target** (str): 
				The column name of the target in the dataset.

		:return: list [best split value, gini_index for this split value]
		:rtype: list
        """
		split_values = dataset[feature]
		#print split_values.shape[0]
		gini_values = np.empty(split_values.shape[0])
		gini_values.fill(1) # fill with worst case

		index = 0
		for i, val in split_values.iteritems():
			g1 = dataset[dataset[feature] < val][target].value_counts()
			g2 = dataset[dataset[feature] >= val][target].value_counts()
			gini_values[index] = self._gini_index([g1,g2])
			index += 1

		return [split_values[gini_values.argmin()], gini_values.min()]

	def _gini_index(self, array_class_counts):
		"""
		Returns the gini-index for the split.

		:paramaters:
			**array_class_count** (list of Pandas Series): 
				Each entry in the array contains a pandas series of the the counts in each class
				for each grouped data points for the split.
				[ series1: (number belong to class 0, number belonging to class 1) , 
				  series2: (number belong to class 0, umber belonging to class 1) 
				 ]

		:return: gini-index of the split
		:rtype: float
		"""
		gini = 0
		for g in array_class_counts:
			tot_in_group = float(g.sum())
			for i, count_in_class in g.iteritems():
				p = float(count_in_class)/ tot_in_group
				gini += p*(1-p)

		return gini


	def _split(self, curr, dataset, target,	depth):
		"""
		Recursive splitting function that creates child
		splits for a node or make this node a terminal node.

		:parameters:
			**curr** (TreeNode): 
				The current node in the Tree

			**dataset** (`DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_ ):
				Training data.

			**target** (str): 
				The column name of the target in the dataset.

			**depth** (int):
				The depth of node curr.
		
		"""
		left_df = dataset[dataset[curr.val['splitting_feature']] < curr.val['splitting_value']]
		right_df = dataset[dataset[curr.val['splitting_feature']] >= curr.val['splitting_value']]

		# check if either split dataset is empty
		if left_df.empty or right_df.empty:
			curr.val = self._make_leaf(dataset[target])

		# deal with tree being at max_depth
		elif depth >= self.max_depth:
			del(dataset)
			curr.left = TreeNode(self._make_leaf(left_df[target]))
			curr.right = TreeNode(self._make_leaf(right_df[target]))
		else:
			del(dataset)
			# process right child
			if left_df.shape[0] <= self.min_size:
				curr.left = TreeNode(self._make_leaf(left_df[target]))
			else:
				curr.left = TreeNode(self._get_split(left_df, target))

				self._split(curr.left,
						left_df,
						target,
						depth+1)

			# process right child
			if right_df.shape[0] <= self.min_size:
				curr.right = TreeNode(self._make_leaf(right_df[target]))
			else:
				curr.right = TreeNode(self._get_split(right_df, target))

				self._split(curr.right,
							right_df,
							target,
							depth+1)
		return

	def _make_leaf(self, target_values):
		"""
        Creates a terminal node value by selecting amoung the group that has
        the majority.

        :param: target_value (Pandas Series): The target values of this data.

        :return: The majority of the class.
        :rtype: int
    	"""

		# not sure i need this check
		if len(target_values.unique()) == 1:
			value = target_values.unique()[0]
		else:
			value = target_values.value_counts(normalize=True).argmax()
		

		return value