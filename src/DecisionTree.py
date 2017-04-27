
from random import seed
from random import randrange
from math import sqrt
seed(1)



class DecisionTree (object):

	def __init__(self, max_depth=2, min_size=1):
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
					b_index, b_value, b_score, b_groups = index, row[index], gini, groups
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
				print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
				self.print_tree(node['left'], depth+1)
				self.print_tree(node['right'], depth+1)
			else:
				print('%s[%s]' % ((depth*' ', node)))


class DecisionTreeClassifier (DecisionTree):

	def __init__(self, max_depth=2, min_size=2, cost='gini'):
		DecisionTree.__init__(self, max_depth, min_size)
		self.class_values = None
		self.cost_function = cost

	def fit(self, train, n_features):
		self.class_values = list(set(row[-1] for row in train))
		self._fit(train, n_features)

	def predict(self, row):
		return self._predict(row, self.root)

	def _cost(self, groups):
		return self._gini_index(groups, self.class_values)

	def _gini_index(self, groups, class_values):
		gini = 0.0
		for class_value in class_values:
			for group in groups:
				size = len(group)
				if size == 0:
					continue
				p = [row[-1] for row in group].count(class_value) / float(size)
				gini += (p * (1.0 - p))
				return gini

	def _make_leaf(self, group):
		outcomes = [row[-1] for row in group]
		return max(set(outcomes), key=outcomes.count)


class RandomForestClassifier:

	def __init__(self, n_trees=10, max_depth=2, min_size=2, cost='gini'):
		self.max_depth = max_depth
		self.min_size = min_size 
		self.cost_function = cost
		self.n_trees = n_trees
		self.trees = list()

	def fit(self, train, test):
		n_features = int(sqrt(len(train[0])-1))

		for i in range(self.n_trees):
			sample = self._subsample(train)
			tree = DecisionTreeClassifier(self.max_depth,
										   self.min_size,
										   self.cost_function)

			tree.fit(sample, n_features)
			self.trees.append(tree)

		predictions = [self.bagging_predict(row) for row in test]
		return(predictions)

	def bagging_predict(self, row):
		predictions = [tree.predict(row) for tree in self.trees]
		return max(set(predictions), key=predictions.count)

	def KFoldCV(self, dataset, n_folds=10):
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

	def _cross_validation_split(self, dataset, n_folds):
		dataset_split = list()
		dataset_copy = list(dataset)
		fold_size = int(len(dataset) / n_folds)
		for i in range(n_folds):
			fold = list()
			while len(fold) < fold_size:
				index = randrange(len(dataset_copy))
				fold.append(dataset_copy.pop(index))
			dataset_split.append(fold)
		return dataset_split

	def _metric(self, actual, predicted):
		return self._accuracy(actual, predicted)

	def _accuracy(self, actual, predicted):
		correct = 0
		for i in range(len(actual)):
			if actual[i] == predicted[i]:
				correct += 1
		return correct / float(len(actual)) * 100.0

	def _subsample(self, dataset):
		sample = list()
		n_sample = round(len(dataset))
		while len(sample) < n_sample:
			index = randrange(len(dataset))
			sample.append(dataset[index])
		return sample


