import numpy as np

class RandomForest:
    """
    A Random Forest base class.

    Classification and Regression Random Forests will be derived classes
    that override certain functions of this class. This was done
    because many common methods, so to reduce code they are written here
    in the base class.

    Parameters
    ------------
    max_depth int: default=2
        The maximum depth of tree.

    min_size int: default=1
        The minimum number of datapoints in terminal nodes.

    n_features int: min_size=None
        The number of features to be used in splitting.

    Attributes
    -----------
    root dictionary dict:
        The root of the decision tree.

    """

    def __init__(self, n_trees=10, max_depth=2, min_size=1):
        self.max_depth = max_depth
        self.min_size  = min_size
        self.n_trees   = n_trees
        self.trees     = None

    def _subsample(self, dataset : np.ndarray) -> np.ndarray:
        """
        This function returns a bootstrapped version of the dataset which
        has the same number of rows.

        Parmeters
        ---------
            dataset np.ndarray : The dataset.

        Returns
        -------
            list. Bootstrapped version of the dataset.
        """

        number_of_rows = dataset.shape[0]
        sample_of_rows = number_of_rows
        random_indices = np.random.choice(number_of_rows,
                                          size=sample_of_rows,
                                          replace=True)
        return dataset[random_indices,:]

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


    def get_params(self, deep=True):
        return {"max_depth" : self.max_depth,
                "min_size"  : self.min_size,
                "cost"      : self.cost,
                "n_trees"   : self.n_trees}

