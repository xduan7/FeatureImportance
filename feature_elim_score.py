""" 
    File Name:          feature_elim_score.py 
    Project Name:       FeatureImportance
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               6/24/2018
    Python Version:     3.6.4
    File Description:   

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier


def feature_elim_on_model(x, y, model, random_state):

    scores = np.zeros(shape=(x.shape[1],))

    for i in range(x.shape[1]):

        x_ = np.delete(x, i, axis=1)
        scores[i] = np.mean(cross_val_score(model, x_, y, cv=5))

    return scores


def feature_elim_score(data: dict,
                       target: np.ndarray,
                       params: dict,
                       model_names: list = None,
                       random_state: int = 0,
                       show_img: bool = False):

    if model_names is None:
        model_names = [
            'Decision Tree Classifier',
            'Random Forest Classifier',
            'Nearest Neighbor Classifier',
            'Linear Support Vector Classifier',
            'RBF Support Vector Classifier', ]

    plt.figure(figsize=(8 * len(data), 6 * len(model_names)))

    for model_idx, model_name in enumerate(model_names):

        for data_idx, (data_name, x) in enumerate(data.items()):

            model_str = model_name + ' on ' + data_name

            # Build a model with the best hyper parameters
            if model_name == 'Decision Tree Classifier':
                model = DecisionTreeClassifier(random_state=random_state,
                                               **params[model_str])
            elif model_name == 'Random Forest Classifier':
                model = RandomForestClassifier(random_state=random_state,
                                               **params[model_str])
            elif model_name == 'Nearest Neighbor Classifier':
                model = KNeighborsClassifier(algorithm='brute',
                                             **params[model_str])
            elif model_name == 'Linear Support Vector Classifier':
                model = LinearSVC(dual=False,
                                  random_state=random_state,
                                  max_iter=1e5,
                                  **params[model_str])
            else:
                model = SVC(random_state=random_state, max_iter=1e5,
                            **params[model_str])

            scores = feature_elim_on_model(x, target, model, random_state)
            indices = np.argsort(scores)

            plt.subplot(len(model_names),
                        len(data),
                        model_idx * 2 + data_idx + 1)
            plt.title('Feature Elimination Scores using %s\non %s)'
                      % (model_name, data_name))
            plt.bar(range(x.shape[1]),
                    scores[indices],
                    edgecolor='k',
                    color='w',
                    align='center')
            plt.xticks(range(x.shape[1]), indices)
            plt.xlim([-1, x.shape[1]])
            plt.ylim([np.min(scores) - 0.01, np.max(scores) + 0.01])

    plt.savefig('./img/feature_elim_score.png')
    if show_img:
        plt.show()
