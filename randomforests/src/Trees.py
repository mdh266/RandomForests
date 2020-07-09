
from random import randrange
import pandas as pd
import numpy as np
import math

from src.utils import _make_dataset


class DecisionTree:
  """
  A decision tree base class. 

  Classification and Regression Trees will be derived class that override 
  certain functions of this class.  This was done because many common
  methods, so to reduce code they are written here in the base class.

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

    if n_features is not None:
      self.n_features = n_features-1
    else:
      self.n_features = None

    self.root         = None


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
        
    if self.n_features is None:
      self.n_features = len(dataset) - 1
            
    # perform optimal split for the root
    self.root = self._get_split(dataset)

    # now recurisively split the roots dataset until the stopping
    # criteria is met.
    root = self._split(self.root, 1)


  def _test_split(
    self, 
    dataset : np.ndarray, 
    column  : int, 
    value   : float
  ) -> tuple:
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


  def _get_split(self, dataset):
    """
    Select the best splitting point and feature for a dataset 
    using a random selection of self.n_features number of features.

    Args:  
      dataset (list of list): Training data.
      
    Returns:
      Dictionary of the best splitting feature of randomly chosen and 
      the best splitting value.
    """
    b_index, b_value, b_score, b_groups = 999, 999, 999, None

    # the features to test among the split
    features = set()

    # randomily select features to consider
    while len(features) < self.n_features:
      index = randrange(len(dataset[0])-1)
      features.add(index)

    # loop through the number of features and values of the data
    # to figure out which gives the best split according
    # to the derived classes cost function value of the tested 
    # split
    for index in features:
      for row in dataset:
        groups = self._test_split(dataset, index, row[index])
        gini   = self._cost(groups)
        if gini < b_score:
          b_index  = index
          b_value  = row[index]
          b_score  = gini
          b_groups = groups

    return {'index':b_index, 'value':b_value, 'groups':b_groups}

  def _split(self, node, depth):
    """
    Recursive splitting function that creates child
    splits for a node or make this node a leaf.
    Note: Leaves are just a value, which is determined
    in the derived class.

    Args:
      node (dictionary): The current node in the tree.

      depth (int) : The depth of node curr.

    Returns: None
    """
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
      node['left'] = node['right'] = self._make_leaf(left + right)
      return
    # check for max depth
    if depth >= self.max_depth:
      node['left'] = self._make_leaf(left)
      node['right'] = self._make_leaf(right)
      return
    # process left child
    if len(left) <= self.min_size:
      node['left'] = self._make_leaf(left)
    else:
      node['left'] = self._get_split(left)
      self._split(node['left'], depth+1)
    # process right child
    if len(right) <= self.min_size:
      node['right'] = self._make_leaf(right)
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
      row list  : 
        The data point to classify.

      node dict  : 
        he current node in the tree.

    Returns
    --------
      The leaf value of this data point.
    """
    if row[node['index']] < node['value']:
      if isinstance(node['left'], dict):
        return self._predict(row, node['left'])
      else:
        return node['left']
    else:
      if isinstance(node['right'], dict):
        return self._predict(row, node['right'])
      else:
        return node['right']

  
class DecisionTreeClassifier (DecisionTree):
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
      self._cost = self._gini_index
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
    self._fit(train, target)

  def predict(self, row):
    """
    Predict the class that this sample datapoint belongs to.

    Parameters
    ------------
    row Pandas Serieshttp://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html>`_) : 
      The datapoint to classify.

    Returns
    --------

    The class the data points belong to.
    """
    if isinstance(row, np.ndarray) is False:
      return self._predict(row.values, self.root)
    else:
      return self._predict(row, self.root)
  

  def _gini_index(self, y : np.ndarray) -> float:

      gini = 1.0
      y_t  = y.reshape(len(y))

      target_val_cts = dict(zip(*np.unique(y_t, return_counts=True)))
      size           = len(y)

      for target_class in target_val_cts:
          p = target_val_cts[target_class] / size
          gini -= p ** 2
          
      return gini


  def _make_leaf(self, y : np.ndarray) -> float :
      from scipy.stats import mode
      y_t  = y.reshape(len(y))

      return mode(y_t)[0][0]




