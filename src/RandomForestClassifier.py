from math import sqrt
from RandomForest import RandomForest
from DecisionTreeClassifier import DecisionTreeClassifier

class RandomForestClassifier (RandomForest):

	def __init__(self, n_trees=10, max_depth=2, min_size=2, cost='gini'):
		RandomForest.__init__(self, cost,  n_trees=10, max_depth=2, min_size=2)
		

	def fit(self, train, test):
		n_features = int(sqrt(len(train[0])-1))

		for i in range(self.n_trees):
			sample = self._subsample(train)
			tree = DecisionTreeClassifier(self.max_depth,
										   self.min_size,
										   self.cost_function)

			tree.fit(sample, n_features)
			self.trees.append(tree)

		predictions = [self.predict(row) for row in test]
		return(predictions)


	def predict(self, row):
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


	def _metric(self, actual, predicted):
		return self._accuracy(actual, predicted)

	def _accuracy(self, actual, predicted):
		correct = 0
		for i in range(len(actual)):
			if actual[i] == predicted[i]:
				correct += 1
		return correct / float(len(actual)) * 100.0


