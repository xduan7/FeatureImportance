""" 
    File Name:          main.py 
    Project Name:       FeatureImportance
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               6/24/2018
    Python Version:     3.6.4
    File Description:   

"""

from get_data import get_data, RAND_SEED
from univariate_score import univariate_score
from grid_search import grid_search
from tree_score import tree_score
from feature_elim_score import feature_elim_score
from feature_rplc_score import feature_rplc_score
from indep_tree_score import indep_tree_score
from deep_explain import deep_explain


if __name__ == '__main__':

    original_x, normalized_x, y = get_data()

    data = {'Original Data': original_x,
            'Normalized Data': normalized_x}

    univariate_score(data=data, target=y)

    best_params = grid_search(data=data,
                              target=y,
                              random_state=RAND_SEED)

    # tree_score(data=data,
    #            target=y,
    #            params=best_params,
    #            random_state=RAND_SEED)

    for n in [20]:
        feature_rplc_score(data=data,
                           target=y,
                           params=best_params,
                           n_components=n,
                           trained=False,
                           random_state=RAND_SEED)

    # feature_rplc_score(data=data,
    #                    target=y,
    #                    params=best_params,
    #                    trained=True,
    #                    random_state=RAND_SEED)

    # indep_tree_score(data=data,
    #                  target=y,
    #                  random_state=RAND_SEED)

    # deep_explain(data=data,
    #              target=y,
    #              num_features=20,
    #              random_state=RAND_SEED)


