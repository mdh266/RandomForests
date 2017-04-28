from random import randrange

class DecisionTree (object):
	"""
	A decision tree base class. Classification and Regression
	Trees will be derived class that override certain functions 
	of this class.  This was done because many common methods,
	so to reduce code they are written here in the base class.
		
	:Attributes: 
		**max_depth** (int): 
			The maximum depth of tree.

		**min_size** (int):
			The minimum number of datapoints in terminal nodes.

		**n_features** (int): 
			The number of randomly chosen features to be used in splitting.

		**root** (dictionary): 
			The root of the decision tree.
	"""
	def __init__(self, max_depth=2, min_size=1):
		"""
		Constructor for a generic decision tree.

		Args: 
			max_depth (int) : The maximum depth of tree.
			min_size (int) : The min number of datapoints in terminal nodes.
		"""
		self.max_depth = max_depth
		self.min_size = min_size
		self.n_features = None
		self.root = None

	def _fit(self, train, n_features=None):

		if n_features is None:
			self.n_features = len(train[0])-1
		else:
			self.n_features = n_features

		self.root = self._get_split(train)

		root = self._split(self.root, 1)


	def _get_split(self, dataset):
		b_index, b_value, b_score, b_groups = 999, 999, 999, None
		features = list()
		while len(features) < self.n_features:
			index = randrange(len(dataset[0])-1)
			if index not in features:
				features.append(index)
		for index in features:
			for row in dataset:
				groups = self._test_split(index, row[index], dataset)
				gini = self._cost(groups)
				if gini < b_score:
					b_index = index
					b_value = row[index]
					b_score = gini
					b_groups = groups
		return {'index':b_index, 'value':b_value, 'groups':b_groups}

	def _test_split(self, index, value, dataset):
		left, right = list(), list()
		for row in dataset:
			if row[index] < value:
				left.append(row)
			else:
				right.append(row)
		return left, right

	# Create child splits for a node or make terminal
	def _split(self, node, depth):
		"""
		Recursive splitting function that creates child
		splits for a node or make this node a leaf.
		Note: Leaves are just a value, which is determined
		in the derived class.

		Args:
			node (dictionary): The current node in the tree.

			depth (int) : The depth of node curr.
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









