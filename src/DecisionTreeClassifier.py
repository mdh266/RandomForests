
from DecisionTree import DecisionTree

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