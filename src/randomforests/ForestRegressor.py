from randomforests.utils import _make_dataset
from randomforests.Forest import RandomForest
from randomforests.TreeRegressor import DecisionTreeRegressor

from math import sqrt
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_squared_error


class RandomForestRegressor(BaseEstimator, ClassifierMixin, RandomForest):
    """
    A random forest regression model that extends the abstract base class
    of random forest.

    Attributes
    ----------
      max_depth int :
        The maximum depth of tree.

      min_size int :
        The minimum number of datapoints in terminal nodes.

      n_features int :
        The number of features to be used in splitting.

      n_trees:
        The number of trees in the forest

      cost str :
        The cost function
    """

    def __init__(
        self,
        n_trees: int = 10,
        max_depth: int = 2,
        min_size: int = 1,
        cost: str = "mse",
    ):
        """
        Constructor for random forest regressor. This mainly just initialize
        the attributes of the class by calling the base class constructor.
        However, here is where it is the cost function string is checked
        to make sure it either using 'mse', otherwise an error is thrown.

        """
        super().__init__(n_trees=n_trees, max_depth=max_depth, min_size=min_size)

        if cost == "mse":
            self.cost = "mse"
        else:
            raise NameError("Not valid cost function")

    def fit(self, X, y=None):
        """
        Fit the random forest to the training set train.

        Note: Below we set the number of features to use in the splitting to be
        the square root of the number of total features in the dataset.

        Parameters
        ----------
        X DataFrame : The feature dataframe or numpy array of features

        y Series : The target variables values

        Returns
        -------

        Fitted model
        """

        n_features = round(sqrt(X.shape[1]))
        dataset = _make_dataset(X, y)
        self.trees = [
            self._bootstrap_tree(dataset=dataset, n_features=n_features)
            for i in range(self.n_trees)
        ]

        return self

    def predict(self, x: pd.DataFrame) -> int:
        """
        Predict the value for this sample datapoint

        Parameters
        ----------
        x  np.ndarray:
          The datapoints to classify.

        Returns
        --------
          The predicted class the data points belong to.
        """
        if isinstance(x, np.ndarray) is False:
            rows = x.to_numpy()
        else:
            rows = x

        preds = np.vstack([tree.predict(rows) for tree in self.trees])

        return np.mean(preds, axis=0)

    def score(self, X=None, y=None):
        """
        Returns the mean squared error of the model

        Parameters
        ----------
        X DataFrame : The feature dataframe

        y Series : The target variables values

        Returns
        -------
        float
        """

        return mean_squared_error(y, self.predict(X))

    def _bootstrap_tree(
        self, dataset: np.ndarray, n_features: int
    ) -> DecisionTreeRegressor:

        sample = self._subsample(dataset)
        tree = DecisionTreeRegressor(
            max_depth=self.max_depth, min_size=self.min_size, n_features=n_features
        )
        return tree.fit(sample[:, :-1], sample[:, -1])
