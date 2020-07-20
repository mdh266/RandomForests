from randomforests.utils import _make_dataset
from randomforests.Forest import RandomForest
from randomforests.TreeClassifier import DecisionTreeClassifier

from math import sqrt
import pandas as pd
import numpy as np
import scipy as sp

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

class RandomForestClassifier (BaseEstimator, ClassifierMixin, RandomForest):

    def __init__(self, n_trees : int = 10, max_depth : int =2, min_size : int =1, cost : str ='gini'):
        """
        Constructor for random forest classifier. This mainly just initialize
        the attributes of the class by calling the base class constructor.
        However, here is where it is the cost function string is checked
        to make sure it either using 'gini', otherwise an error is thrown.

        """
        super().__init__(n_trees   = n_trees,
                         max_depth = max_depth,
                         min_size  = min_size)

        if cost == 'gini':
            self.cost  = "gini"
        else:
            raise NameError('Not valid cost function')



    def fit(self, X, y = None):
        """
        Fit the random forest to the training set train.  If a test set is provided
        then the return value wil be the predictions of the RandomForest on the
        test set.  If no test set is provide nothing is returned.


        Note: Below we set the number of features to use in the splitting to be
        the square root of the number of total features in the dataset.

        Parameters
        -----------
        """

        n_features = round(sqrt(X.shape[1]))
        dataset    = _make_dataset(X,y)
        self.trees = [self._bootstrap_tree(dataset    = dataset,
                                           n_features = n_features)
                      for i in range(self.n_trees)]

        return self


    def predict(self, rows : pd.DataFrame) -> int:
        """
        Predict the class that this sample datapoint belongs to.

        Parameters
        ----------
        rows  np.ndarray:
          The datapoints to classify.

        Returns
        --------
          The predicted class the data points belong to.
        """
        if isinstance(rows, np.ndarray) is False:
            x = rows.to_numpy()
        else:
            x = rows

        preds = [tree.predict(x) for tree in self.trees]

        return sp.stats.mode(preds)[0][0]


    def score(self, X=None, y=None):
        """
        Returns the accuracy of the model

        Parameters
        ----------
        X DataFrame : The feature dataframe

        y Series : The target variables values

        """

        return accuracy_score(self.predict(X),y)

    def _bootstrap_tree(self, dataset : np.ndarray, n_features : int) -> DecisionTreeClassifier:

        sample = self._subsample(dataset)
        tree   = DecisionTreeClassifier(max_depth  = self.max_depth,
                                        min_size   = self.min_size,
                                        n_features = n_features)
        return tree.fit(sample[:,:-1],sample[:,-1])