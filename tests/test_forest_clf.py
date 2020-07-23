import pytest
import numpy as np
from randomforests.utils import _make_dataset
# from randomforests.TreeClassifier import DecisionTreeClassifier
from randomforests.ForestClassifier import RandomForestClassifier


def test_default():
    forest = RandomForestClassifier()

    assert (forest.n_trees   == 10 and
            forest.max_depth == 2 and
            forest.min_size  == 1 and
            forest.cost      == 'gini')


params_tests = [
    ({"max_depth":3, "min_size":5,  "n_trees":53, "cost":'gini'},
     {"max_depth":3, "min_size":5,  "n_trees":53, "cost":'gini'})
]

@pytest.mark.parametrize('test_dict, expected_dict', params_tests)
def test_get_params(test_dict, expected_dict):
    forest = RandomForestClassifier(max_depth = test_dict["max_depth"],
                                    min_size  = test_dict["min_size"],
                                    n_trees   = test_dict["n_trees"])

    assert expected_dict == forest.get_params()


def test_make_bootsrap():
    """
    Cant really do good test since it has random sample with replacement
    """
    X = np.array([[0.1],
                  [0.5],
                  [0.7],
                  [0.9]])

    y = np.array([0, 0, 1, 1])

    dataset = _make_dataset(X,y)

    forest = RandomForestClassifier()
    tree   = forest._bootstrap_tree(dataset=dataset, n_features=1)

    assert tree.n_features == 1

def test_fit():
    """
    Cant really do good test since it has random sample with replacement
    """
    X = np.array([[0.1],
                  [0.5],
                  [0.7],
                  [0.9]])

    y = np.array([0, 0, 1, 1])

    dataset = _make_dataset(X,y)

    forest = RandomForestClassifier()
    model  = forest.fit(X,y)

    assert len(model.trees) == 10


# def test_predict():
#     """
#     Cant really do good test since it has random sample with replacement
#     """
#     X = np.array([[0.1],
#                   [0.5],
#                   [0.7],
#                   [0.9]])
#
#     y = np.array([0, 0, 1, 1)
#
#     dataset = _make_dataset(X,y)
#
#     forest = RandomForestClassifier()
#     model  = forest.fit(X,y)
#
#     assert len(model.trees) == 10

