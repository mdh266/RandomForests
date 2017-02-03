from random import randrange  
import pandas as pd

from DecisionTree import DecisionTree
from DecisionTree import TreeNode

class DecisionTreeClassifier (DecisionTree):
	
	def __init__(self, max_depth=2, min_size=5):
		"""
		Constructor for a classification decision tree.

		:param int max_depth: The maximum depth of tree.
		:param int min_size: The minimum number of datapoints in terminal nodes.
		"""
		DecisionTree.__init__(self, max_depth, min_size)
	
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
		Select the best split point for a dataset using a random Selection of the features.

		:parameters:

 			**dataset** (`DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_ ):
				Training data.
			
			**target** (str): 
				The column name of the target in the dataset.

		:rtype: dict
        	"""
		best_feature, best_value, best_score, best_groups = 999,999,999,None

		# the features to test among the split
		features = list()
		cols = dataset.columns

		# randomily select features to consider 
		while len(features) < self.n_features:

			# out of all the features
			feature_col = randrange(dataset.shape[1]-1)
 
			# exclude the target as a possible feature col
			if cols[feature_col] not in features and cols[feature_col] != target:
				features.append(cols[feature_col])
		
		# loop through the number of features to figure out which
		# gives the best split.
		for feature in features:
			# split the data set according to this feature
			# and find the splits gini_index
			feature_values = dataset[feature]
			for index_val_pair in feature_values.iteritems():

				left_group, right_group = self._test_split(feature, 
										 	index_val_pair[1], 
										  	dataset)

				gini = self._gini_index(left_group[target], right_group[target])
			
				# if this is the best split update the info
				if gini < best_score:
					best_feature = feature
					best_value = index_val_pair[1]
					best_score = gini
					#best_groups = [left_group, right_group]

		return {'splitting_feature': best_feature,
                'splitting_value': best_value}



	


	def _gini_index(self, target_group_1, target_group_2):
		"""
		Calculates the Gini index for a split dataset by counting up
		the percentages of each targets classes in the split groups.
	
		:param Series target_group_1: The Panda Series of target values in
			 group 1 after splitting.
		:param Series target_group_1: The Panda Series of target values in
			 group 2 after splitting.
		
		:return: Returns the gini-index
		:rtype: float
		"""

		gini = 0.0
		
		if target_group_1.empty is False:
			# get the percentage of class types in the first group
			group_1_percentages = target_group_1.value_counts(normalize=True)

			# loop over and get the percentages of the class in this group
			for class_percent_tupple in group_1_percentages.iteritems():
				percentage = class_percent_tupple[1]
				
				#update gini index
				gini += percentage * (1 -percentage)
		
		if target_group_2.empty is False:
		# get the percentage of class types in the first group
			group_2_percentages = target_group_2.value_counts(normalize=True) 
			
			# loop over and get the percentages of the class in this gruop
			for class_percent_tupple in group_2_percentages.iteritems():
				percentage = class_percent_tupple[1]

				#update gini index
				gini += percentage * (1 -percentage)
	
		return gini


	def _split(self, curr, dataset, target,	depth):
		"""
		Recursive splitting function that creates child
		splits for a node or make this node a terminal node.
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
			if left_df.shape[0] <= self.min_size:
				curr.left = TreeNode(self._make_leaf(left_df[target]))
			else:
				curr.left = TreeNode(self._get_split(left_df, target))

				self._split(curr.left,
						left_df,
						target,
						depth+1)

			# process left child
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
    
        	Input:
       		@node - of the decision tree
    		@target_values.
    		"""

		# not sure i need this check
		if len(target_values.unique()) == 1:
			value = target_values.unique()[0]
		else:
			value = target_values.value_counts(normalize=True).argmax()
		

		return value