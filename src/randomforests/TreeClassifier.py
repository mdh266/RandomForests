import numpy as np
import scipy as sp

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

from randomforests.Tree import DecisionTree

class DecisionTreeClassifier (BaseEstimator, ClassifierMixin, DecisionTree ):
    """
    A decision tree classifier that extends the DecisionTree class.

    Attributes
    ----------
      max_depth int :
        The maximum depth of tree.

      min_size int :
        The minimum number of datapoints in terminal nodes.

      n_features int :
        The number of features to be used in splitting.

      root dict :
        The root of the decision tree.

      cost str :
        The cost function
    """

    def __init__(self, max_depth=2, min_size=1, n_features = None, cost='gini'):

        super().__init__(max_depth  = max_depth,
                         min_size   = min_size,
                         n_features = n_features)

        if cost == 'gini':
            self.cost  = "gini"
            self._cost = self._cost_gini
        else:
            raise NameError('Not valid cost function')

    def fit(self, X=None, y=None):
        """
        Builds the classification decsision tree by recursively splitting
        tree until the the maxmimum depth, max_depth of the tree is acheived or
        the node have the minimum number of training points, min_size.

        n_features will be passed by the RandomForest as it is usually a subset
        of the total number of features. However, if one is using the class as a
        stand alone decision tree, then the n_features will automatically be

        Parameters
        ----------
        X DataFrame : The feature dataframe

        y Series : The target variables values

        """
        self._fit(X, y)

        return self

    def score(self, X=None, y=None):
        """
        Returns the accuracy of the model

        Parameters
        ----------
        X DataFrame : The feature dataframe

        y Series : The target variables values

        """

        return accuracy_score(self.predict(X),y)


    def _cost_gini(self, groups : tuple) -> float:
        """
        Get the cost of the spit of the dataframe. Groups
        will be the tuple containing the left and right
        splits.

        Parameters
        -----------
        groups tuple :  The left and right versions of the dataset
          after the split

        Returns
        -------
        float : The cost of the split.

        """
        cost = 0.0
        for group in groups:
            cost += self._gini_index(group[:,-1])

        return cost


    def _gini_index(self, y : np.ndarray) -> float:
        """
        Gini index for a single target vector.
        """
        gini = 0.0
        y_t  = y.reshape(len(y))

        target_val_cts = dict(zip(*np.unique(y_t, return_counts=True)))
        size           = len(y)

        if size != 0:
            for target_class in target_val_cts:
                p = target_val_cts[target_class] / size
                gini += p * (1 - p)

        return gini

    def _make_leaf(self, y : np.ndarray) -> float :
        """
        Makest the leaf of the tree by taking the value of the class
        that has the largest size.

        Parameters
        ----------
        y np.ndarray : The target classes

        Returns
        -------
        The leaf value.

        """
        y_t  = y.reshape(len(y))

        return sp.stats.mode(y_t)[0][0]