import numpy as np

from randomforests.Forest import RandomForest


def test_init():
    forest = RandomForest()
    assert forest.n_trees == 10 and forest.min_size == 1 and forest.max_depth == 2


def test_subsample():
    forest = RandomForest()
    dataset = np.array([[0.1, 0], [0.5, 0], [0.7, 1], [0.9, 1]])
    resampled = forest._subsample(dataset)
    assert resampled.shape == dataset.shape
