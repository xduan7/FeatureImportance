""" 
    File Name:          feature_rplc_score.py 
    Project Name:       FeatureImportance
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               6/24/2018
    Python Version:     3.6.4
    File Description:   

"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import FastICA, PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier


def feature_rplc_on_model(x_train, x_test, y_train, y_test,
                          untrained_model, trained_model):

    num_samples, num_features = x_train.shape

    scores = np.zeros(shape=(num_features, ))
    for i in range(num_features):

        x_test_ = x_test.copy()
        x_test_[:, i] = 0.

        if trained_model is not None:
            scores[i] = trained_model.score(x_test_, y_test)
        else:
            x_train_ = x_train.copy()
            x_train_[:, i] = 0.
            tmp_model = deepcopy(untrained_model)
            tmp_model.fit(x_train_, y_train)
            scores[i] = tmp_model.score(x_test_, y_test)

    return scores


def feature_rplc_score(data: dict,
                       target: np.ndarray,
                       params: dict,
                       trained: bool,
                       n_components: int = 20,
                       model_names: list = None,
                       random_state: int = 0,
                       show_img: bool = False):

    np.random.seed(random_state)

    if model_names is None:
        model_names = [
            'Decision Tree Classifier',
            'Random Forest Classifier',
            'Nearest Neighbor Classifier',
            'Linear Support Vector Classifier',
            'RBF Support Vector Classifier']

    plt.figure(figsize=(8 * len(data), 6 * len(model_names)))

    for model_idx, model_name in enumerate(model_names):

        for data_idx, (data_name, x) in enumerate(data.items()):



            model_str = model_name + ' on ' + data_name

            # Build a model with the best hyper parameters
            if model_name == 'Decision Tree Classifier':
                untrained_model = DecisionTreeClassifier(
                    random_state=random_state,
                    **params[model_str])
            elif model_name == 'Random Forest Classifier':
                untrained_model = RandomForestClassifier(
                    random_state=random_state,
                    **params[model_str])
            elif model_name == 'Nearest Neighbor Classifier':
                untrained_model = KNeighborsClassifier(
                    algorithm='brute', **params[model_str])
            elif model_name == 'Linear Support Vector Classifier':
                untrained_model = LinearSVC(
                    dual=False,
                    random_state=random_state,
                    max_iter=1e5,
                    **params[model_str])
            elif model_name == 'RBF Support Vector Classifier':
                untrained_model = SVC(
                    random_state=random_state,
                    max_iter=1e5,
                    **params[model_str])
            else:
                untrained_model = None

            x_train, x_test, y_train, y_test = \
                train_test_split(x, target, random_state=random_state)

            if n_components != 20:
                pca = PCA(n_components=n_components)
                x_train_ = pca.fit_transform(x_train)
                x_test_ = pca.transform(x_test)
            else:
                x_train_ = x_train
                x_test_ = x_test

            if trained:
                trained_model = deepcopy(untrained_model)
                trained_model.fit(x_train, y_train)
            else:
                trained_model = None

            scores = feature_rplc_on_model(
                x_train_, x_test_, y_train, y_test,
                untrained_model, trained_model)

            if n_components != 20:
                scores = pca.inverse_transform(scores)
                # scores = np.abs(scores)

            indices = np.argsort(scores)

            plt.subplot(len(model_names),
                        len(data),
                        model_idx * 2 + data_idx + 1)
            plt.title('Feature Replacement Scores using %s\non %s)'
                      % (model_name, data_name))
            plt.bar(range(x.shape[1]),
                    scores[indices],
                    edgecolor='k',
                    color='w',
                    align='center')
            plt.xticks(range(x.shape[1]), indices)
            plt.xlim([-1, x.shape[1]])
            plt.ylim([np.min(scores) - 0.01, np.max(scores) + 0.01])

    if n_components != 20:
        if trained:
            plt.savefig('./img/feature(%d)_rplc(trained)_score.png'
                        % n_components)
        else:
            plt.savefig('./img/feature(%d)_rplc(untrained)_score.png'
                        % n_components)
    else:
        if trained:
            plt.savefig('./img/feature_rplc(trained)_score.png')
        else:
            plt.savefig('./img/feature_rplc(untrained)_score.png')

    if show_img:
        plt.show()


