from random import randrange  
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


	def set_n_features(self, n_features):
		"""
		Set the number of features to choose the split. This could be different
		then the number of features in the data set because you one might
		be using this class from a Random Forest.

		
		:parameters:
			**n_features** (int): 
				The number of features to choose the split.
		"""
		self.n_features = n_features


	def predict(self, row):
		"""
		Predict the class that this datapoint belongs to.

		:parameters:
			 **row** (`Series <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html>`_): 
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
            