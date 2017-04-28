import numpy as np


class DecisionTreeRegressor (DecisionTree):

	def __init__(self, max_depth=2, min_size=2, cost='mse'):
		DecisionTree.__init__(self, max_depth, min_size)
		self.cost_function = cost

	def fit(self, train, n_features):
		self.class_values = list(set(row[-1] for row in train))
		self._fit(train, n_features)

	def predict(self, row):
		return self._predict(row, self.root)

	def _cost(self, groups):
		return self._mse(groups)

	def _mse(self, groups):
		mse = 0.0
		for group in groups:
			if len(group):
					continue
			outcomes = [row[-1] for row in group]
			mse += np.std(outcomes)
		return mse

	def _make_leaf(self, group):
		outcomes = [row[-1] for row in group]
		return np.mean(outcomes)