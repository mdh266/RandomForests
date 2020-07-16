from random import randrange
import pandas as pd
import numpy as np
import math

from src.utils import _make_dataset
from sklearn.base import BaseEstimator, ClassifierMixin


class DecisionTree(BaseEstimator):
  """
  A decision tree abstract base class. 

  Classification and Regression Trees will be derived class that override 
  certain functions of this class, mainly the cost function and make leaf
  function. They will also need a .fit and .predict function.


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
  def __init__(self, max_depth = 2, min_size = 1, n_features = None):

    self.max_depth    = max_depth
    self.min_size     = min_size
    self.n_features   = None

    if n_features is not None:
      self.n_features = n_features

    self.root         = None


  def _set_features(self, X : np.ndarray) -> None:
    """
    Sets the number of features we want use to search for the best split.
    This isn't useful for decision trees specifically, but useful for 
    Random Forests where we don't use all features possible, but use 
    a random subset.


    Parameters
    ----------
    X  np.ndarray The feature dataset

    """
    if self.n_features is None:
      self.n_features = X.shape[1]
    else:
      if (self.n_features != X.shape[-1]):
        raise AttributeError("n_features != X.shape[1]") 


  def _fit(self, X = None, Y = None):
    """
    Builds the decsision tree by recursively splitting tree until the
    the maxmimum depth, max_depth, of the tree is acheived or the nodes
    have the minmum number of training points per node, min_size, is
    achieved.

    Note: n_features will be passed by the RandomForest as it is 
        usually ta subset of the total number of features. 
        However, if one is using the class as a stand alone decision
        tree, then the n_features will automatically be 
    
    Parameters
    ----------
      X DataFrame of the features dataset.

      Y Series of the targetvariable
    """
        
    self._set_features(X)
    
    dataset  = _make_dataset(X = X, y = y)
            
    # perform optimal split for the root
    self.root = self._get_split(dataset)

    # now recurisively split the roots dataset until the stopping
    # criteria is met.
    root = self._split(self.root, 1)



  def _test_split(self, dataset : np.ndarray, column : int, value : float) -> tuple:
    """
    This function splits the data set depending on the feature (index) and
    the splitting value (value)

    Parameters
    -----------
      index   : The column index of the feature.
      value   : The value to split the data.
      dataset : The list of list representation of the dataframe

    Returns
    ---------
      Tupple of the left and right split datasets.

    """
    left  = dataset[dataset[:,column] < value]
    right = dataset[dataset[:,column] >= value]
    return left, right


  def _get_split(self, dataset : np.ndarray) -> dict:
    """
    Select the best splitting point and feature for a dataset 
    using a random subset of self.n_features number of features.

    Parameters
    -----------  
      dataset np.ndarray: 
         Training data.
      
    Returns
    -------
      dict Dictionary of the best splitting feature of randomly chosen and 
      the best splitting value.
    """
    
    b_index, b_value, b_score, b_groups = 999, 999, 999, None

    # the features to test among the split
    features = set()

    # randomily select features to consider
    # TODO: push this to another function or into set_features?
    while len(features) < self.n_features:
      index = randrange(self.n_features)
      features.add(index)

    # loop through the number of features and values of the data
    # to figure out which gives the best split according
    # to the derived classes cost function value of the tested 
    # split
    for column in features:
      for row in dataset[:,column]:
        groups = self._test_split(dataset, column, row)
        gini   = self._cost(groups)
        if gini < b_score:
          b_column  = column
          b_value   = row
          b_score   = gini
          b_groups  = groups


    return {'column':b_column, 'value':b_value, 'groups':b_groups}

  def _split(self, node : dict, depth : int) -> None:
    """
    Recursive splitting function that creates child
    splits for a node or make this node a leaf.
    Note: Leaves are just a value, which is determined
    in the derived class.

    Parameters
    -----------
      node dictionary:
        The current node in the tree.

      depth int : 
        The depth of node curr.

    Returns
    --------]
    """
    left, right = node['groups']
    del(node['groups'])

    # check for a no split in left
    if left.size == 0:
      node['left'] = node['right'] = self._make_leaf(right[:,-1])
      return 
     # check for a no split in right
    elif right.size == 0:
      node['left'] = node['right'] = self._make_leaf(left[:,-1])
      return
    #check for max depth
    elif depth >= self.max_depth:
        node['left'] = self._make_leaf(left[:,-1])
        node['right'] = self._make_leaf(right[:,-1])
        return
    # else 
    else:
        # process left child
        if len(left) <= self.min_size:
          node['left'] = self._make_leaf(left[:,-1])

        else:
          node['left'] = self._get_split(left)
          self._split(node['left'], depth+1)

        # process right child
        if len(right) <= self.min_size:
          node['right'] = self._make_leaf(right[:,-1])

        else:
          node['right'] = self._get_split(right)
          self._split(node['right'], depth+1)


  def _predict(self, row : np.ndarray, node : dict):
    """
    Predicts the target value that this datapoint belongs to by recursively
    traversing tree and returns the termina leaf value corresponding 
    to this data point.

    Parameters
    -----------
      row np.ndarray  : 
        The data point to classify.

      node dict  : 
        he current node in the tree.

    Returns
    --------
      The leaf value of this data point.
    """
    if row[node['column']] < node['value']:
      if isinstance(node['left'], dict):
        return self._predict(row, node['left'])
      else:
        return node['left']
    else:
      if isinstance(node['right'], dict):
        return self._predict(row, node['right'])
      else:
        return node['right']

  
class DecisionTreeClassifier (DecisionTree, ClassifierMixin):
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

    columns list  : 
      The feature names.

    cost_function str : 
      The name of the cost function to use: 'gini'.
  """

  def __init__(self, max_depth=2, min_size=2, n_features = None, cost='gini'):

    super().__init__(max_depth  = max_depth, 
                     min_size   = min_size,
                     n_features = n_features)

    if cost == 'gini':
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

  def predict(self, row : np.ndarray) -> int:
    """
    Predict the class that this sample datapoint belongs to.

    Parameters
    ----------
    row  np.ndarray: 
      The datapoint to classify.

    Returns
    --------
      The predicted class the data points belong to.
    """
    if isinstance(row, np.ndarray) is False:
      return self._predict(row.values, self.root)
    else:
      return self._predict(row, self.root)
  

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
    from scipy.stats import mode
    y_t  = y.reshape(len(y))

    return mode(y_t)[0][0]