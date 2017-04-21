from random import randint 
import pandas as pd


class TreeNode:
	"""
	TreeNode is a class used to build a binary decison tree.

	:Parameters:
		**value** (dict or int): 
			If the node is a leaf, value is the predicted class otherwise it's a dictionary.
		

	:Attributes: 

		**val** (dict or int): 
			If the node is a leaf, val is the predicted class otherwise it's a dictionary.
	
	 	**left** (:class:`TreeNode`): 
			The left child.

		**right** (:class:`TreeNode`): 
			The right child.
		
	"""
	def __init__(self, value):
		"""
		Constructor for TreeNode in the decision tree. 
	
		:param dict or int value: If the node is a leaf, value is the predicted class otherwise it's a dictionary.
		"""
		# dictionary if not terminal node, otherwise it it the class value.
		self.val = value
		
		# left child
		self.left = None

		# right child
		self.right = None


class DecisionTree:
	"""
	A decision tree classifier based off the gini-index.
	
	:Parameters:
		**max_depth** (int):
			The maximum depth of tree.

		**min_size** (int):
			The minimum number of datapoints in terminal nodes.
		
	:Attributes: 
		**max_depth** (int): 
			The maximum depth of tree.

		**min_size** (int):
			The minimum number of datapoints in terminal nodes.

		**original_n_features** (int): 
			The number of features in the dataset.

		**n_features** (int): 
			The number of randomly chosen features to be used in splitting.

		**root** (:class:`TreeNode`): 
			The root of the decision tree.
	"""

	def __init__(self, max_depth=2, min_size=5):
		"""
		Constructor for a classification decision tree.

		:param int max_depth: The maximum depth of tree.
		:param int min_size: The minimum number of datapoints in terminal nodes.
		"""
		self.max_depth = max_depth
		self.min_size = min_size
		self.original_n_features = None
		self.n_features =  None 
		self.root = None
		self.columns = None


	def set_n_features(self, n_features):
		"""
		Set the number of features to choose the split. This could be different
		then the number of features in the data set because you one might
		be using this class from a Random Forest.

		
		:parameters:
			**n_features** (int): 
				The number of features to choose the split.
		"""
		self.n_features = int(n_features)

	def predict(self, row):
		"""
		Predict the class that this datapoint belongs to.

		:param: row (`Series <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html>`_): 
				The datapoint to classify.

		:return: The class the data points belong to.
		:rtype: int
		"""
		
		# check to make sure data point is the right size
		assert row.shape[0] == self.original_n_features
		
		# Go down the tree until get a leaf.
		curr = self.root
		while isinstance(curr.val, dict):
			if row[curr.val['splitting_feature']] < curr.val['splitting_value']:
				curr = curr.left
			else:
				curr = curr.right

		# now at a terminal node and get the predtion
		return curr.val

	def _convert_to_list(self, df, target):
		"""
		Converts list Pandas dataframe to list of lists.

		:param: df (DataFrame): Pandas DataFrame
		:param: target (target): The target name
		:return: list of lists of dataframe
		:rvalue: list
		"""
		# set the column names 
		self.columns = df.columns

		# make new dataset dictionary of lists
		new_dataset = []

		targets = df[target]
		df = df.drop(target,axis=1)

		# loop over each row and convert from Pandas Series to list
		for row in df.iterrows():
			new_dataset.append(row[1].tolist())

		for i in range(len(new_dataset)):
			new_dataset[i].append(targets[i])

		return new_dataset

	def _split(self, curr, dataset,	depth):
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
		left_dataset, right_dataset = self._split_dataset(dataset,
														curr.val['splitting_col'],
														curr.val['splitting_value'])

		# check if either split dataset is empty
		if len(left_dataset) == 0 or len(right_dataset) == 0:
			curr.val = self._make_leaf(dataset)
			del(dataset)
		# deal with tree being at max_depth
		elif depth >= self.max_depth:
			del(dataset)
			curr.left = TreeNode(self._make_leaf(left_dataset))
			curr.right = TreeNode(self._make_leaf(right_dataset))
		else:
			del(dataset)
			# process right child
			if len(left_dataset) <= self.min_size:
				curr.left = TreeNode(self._make_leaf(left_dataset))
			else:
				curr.left = TreeNode(self._get_split(left_dataset))

				self._split(curr.left, left_dataset, depth+1)

			# process right child
			if len(right_dataset) <= self.min_size:
				curr.right = TreeNode(self._make_leaf(right_dataset))
			else:
				curr.right = TreeNode(self._get_split(right_dataset))

				self._split(curr.right,right_dataset ,depth+1)
		return

	def _get_split(self, dataset):
		"""
		Select the best split point and feature for a dataset using a random Selection of the features.

		:param: dataset (`DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_ ):
				Training data.
			
		:param: target (str): 
				The column name of the target in the dataset.

		:return: Dictionary of the best splitting feature of randomly chosen and the best splitting value.
		:rtype: dict
		"""
		best_feature, best_value, best_score, best_groups = 999,999,999,None

		# the features to test among the split
		features = list()

		# randomily select features to consider 
		while len(features) < self.n_features:

			# out of all the features
			feature_col = randint(0, self.n_features-1)
 
			# exclude the target as a possible feature col
			if feature_col not in features:
				features.append(feature_col)
		
		#print features
		# loop through the number of features to figure out which
		# gives the best split.
		for feature in features:
			# split the data set according to this feature
			# and find the splits gini_index
			#print "feature : " + feature
			for row in dataset:
				split_val = row[feature]
				groups = self._split_dataset(dataset, feature, split_val)

				error = self._error(groups)

				# if this is the best split update the info
				if error < best_score:
					best_col = feature
					best_value = split_val
					best_score = error
					#best_groups = [left_group, right_group]

		return {'splitting_col': best_col,
				'splitting_feature': self.columns[best_col],
                'splitting_value': best_value}

	def _split_dataset(self, dataset, feature, value):
		"""
		"""

		left_dataset, right_dataset = list(), list()

		for row in dataset:
			if row[feature] < value:
				left_dataset.append(row)
			else:
				right_dataset.append(row)

		return left_dataset, right_dataset

	def _to_string(self):
		"""
		Prints breadth first traversal of the tree to a string.

		This is used for comparison in testing.

		:return: Returns a string of the breadth first traversal.
		:rtype: str
		"""
		s = ""
		queue = [self.root]
		visited = []
		while len(queue) != 0:
			curr = queue.pop(0)
			if curr not in visited:
				visited.append(curr)
				if curr.left != None:
					queue.append(curr.left)
				if curr.right != None:
					queue.append(curr.right)
		
		for node in visited:
			if isinstance(node.val, dict):
				s += '[ ' + node.val['splitting_feature'] + ' < '\
					+  str(node.val['splitting_value']) + ' ] \n'
			else:
				s += str(node.val) + '\n'

		return s

	def print_tree(self):
		"""
		Prints the tree using a pre-order traversal.
		"""
		self._print_tree(self.root,1)


	def _print_tree(self, node, depth):
		"""
		Inner recursive call for printing the tree.
		
		:param: TreeNode node: The TreeNode node in the binary tree.
		:param: int depth: The depth of this node in the tree.
		"""
		if isinstance(node.val, dict):
			print('%s [%s < %.3f]' % ((depth*' ',
				 node.val['splitting_feature'],
				  node.val['splitting_value'])))

			self._print_tree(node.left, depth+1)
			self._print_tree(node.right, depth+1)
		else:
			print('%s[%s]' % ((depth*' ', node.val)))
            