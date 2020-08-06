import pytest
from randomforests.ForestRegressor import RandomForestRegressor


def test_default():
    forest = RandomForestRegressor()

    assert (forest.n_trees   == 10 and
            forest.max_depth == 2 and
            forest.min_size  == 1 and
            forest.cost      == 'mse')

def test_cost_exception():
    with pytest.raises(Exception):
        forest = RandomForestRegressor(cost="mae")


params_tests = [
    ({"max_depth":3, "min_size":5,  "n_trees":53, "cost":'mse'},
     {"max_depth":3, "min_size":5,  "n_trees":53, "cost":'mse'})
]

@pytest.mark.parametrize('test_dict, expected_dict', params_tests)
def test_get_params(test_dict, expected_dict):
    forest = RandomForestRegressor(max_depth = test_dict["max_depth"],
                                    min_size  = test_dict["min_size"],
                                    n_trees   = test_dict["n_trees"])

    assert expected_dict == forest.get_params()
