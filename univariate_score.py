""" 
    File Name:          univariate_score.py 
    Project Name:       FeatureImportance
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               6/24/2018
    Python Version:     3.6.4
    File Description:   

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import \
    SelectKBest, f_classif, mutual_info_classif


def univariate_score(data: dict,
                     target: np.ndarray,
                     metrics: dict = None,
                     show_img: bool = False):

    if metrics is None:
        metrics = {'f_value': f_classif, 'mutual_info': mutual_info_classif}

    plt.figure(figsize=(8 * len(data), 6 * len(metrics)))

    for metric_idx, (metric_name, metric) in enumerate(metrics.items()):

        for data_idx, (data_name, x) in enumerate(data.items()):

            scores = SelectKBest(metric, k='all').fit(x, target).scores_
            indices = np.argsort(scores)[::-1]

            plt.subplot(len(metrics), len(data), metric_idx * 2 + data_idx + 1)
            plt.title('Univariate Scores using %s on %s' %
                      (metric_name, data_name))
            plt.bar(range(x.shape[1]),
                    scores[indices],
                    edgecolor='k',
                    color='w',
                    align='center')
            plt.xticks(range(x.shape[1]), indices)
            plt.xlim([-1, x.shape[1]])

    plt.savefig('./img/univariate_score.png')
    if show_img:
        plt.show()
