from random import randrange
from random import seed
seed(1)

class RandomForest (object):

	def __init__(self, cost, n_trees=10, max_depth=2, min_size=2):

		self.max_depth = max_depth
		self.min_size = min_size 
		self.cost_function = cost
		self.n_trees = n_trees
		self.trees = list()

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

	def _subsample(self, dataset):
		
		sample = list()
		n_sample = round(len(dataset))
		while len(sample) < n_sample:
			index = randrange(len(dataset))
			sample.append(dataset[index])
		return sample