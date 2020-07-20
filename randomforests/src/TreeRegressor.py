import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_squared_error

from src.Tree import DecisionTree

class DecisionTreeRegressor (BaseEstimator, ClassifierMixin, DecisionTree ):
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

    """

    def __init__(self, max_depth=2, min_size=1, n_features = None):

        super().__init__(max_depth  = max_depth,
                         min_size   = min_size,
                         n_features = n_features)

        self.cost  = "mse"
        self._cost = self._cost_mse


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

        return mean_squared_error(self.predict(X),y)

    def _cost_mse(self, groups : tuple) -> float:
        """
        Get the cost of the spit of the dataframe. Groups will be the tuple
        containing the left and right splits. The cost is the mean square error,
        which is the same as std ** 2.


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
            cost += np.std(group[:,-1]) ** 2

        return cost

    def _make_leaf(self, y : np.ndarray) -> float :
        """
        Makest the leaf of the tree by taking the mean of the target values

        Parameters
        ----------
        y np.ndarray : The target values

        Returns
        -------
        The leaf value.

        """
        y_t  = y.reshape(len(y))

        return np.mean(y_t)
