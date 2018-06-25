""" 
    File Name:          indep_tree_score.py 
    Project Name:       FeatureImportance
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               6/24/2018
    Python Version:     3.6.4
    File Description:   

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA, PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def indep_tree_score(data: dict,
                     target: np.ndarray,
                     model_names: list = None,
                     random_state: int = 0,
                     show_img: bool = False):

    if model_names is None:
        model_names = [
            'Decision Tree Classifier',
            'Random Forest Classifier', ]

    decompositions = {
        'PCA(n=12)': PCA(n_components=12),
        'PCA(n=8)': PCA(n_components=8),
        'PCA(n=4)': PCA(n_components=4),
        'ICA(n=12)': FastICA(n_components=12,
                             max_iter=20000,
                             tol=0.005,
                             random_state=random_state),
        'ICA(n=8)': FastICA(n_components=8,
                            max_iter=20000,
                            tol=0.005,
                            random_state=random_state),
        'ICA(n=4)': FastICA(n_components=4,
                            max_iter=20000,
                            tol=0.005,
                            random_state=random_state), }

    plt.figure(figsize=(8 * len(model_names), 6 * len(decompositions)))

    x = data['Normalized Data']

    for model_idx, (model_name) in enumerate(model_names):

        for decomp_idx, (decomp_name, decomp) in enumerate(decompositions.items()):

            x_train, x_test, y_train, y_test = \
                train_test_split(x, target, random_state=random_state)

            # scores = cross_val_score(model, X, y, cv=10)
            x_train_ = decomp.fit_transform(x_train)
            x_test_ = decomp.transform(x_test)

            # Try and find the best model
            # Build a model with the best hyper parameters
            pipeline = Pipeline(steps=[('clf', None), ])

            if model_name == 'Decision Tree Classifier':

                param_grid = [{
                    'clf':
                        [DecisionTreeClassifier(random_state=random_state)],
                    'clf__criterion':
                        ['gini', 'entropy'],
                    'clf__max_depth':
                        [4, 8, 12, 16, 20, 24, ],
                    'clf__min_samples_leaf':
                        [1, 2, 4, 8, 16, 32, ],
                    'clf__max_features':
                        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ],
                }, ]

                grid_search = GridSearchCV(
                    pipeline,
                    param_grid=param_grid,
                    cv=5,
                    n_jobs=-1,
                    scoring='accuracy')

                grid_search.fit(x_train_, y_train)
                model_best_params = grid_search.best_params_.copy()
                model_best_params.pop('clf')
                # print(model_best_params)

                for key in list(model_best_params):
                    new_key = key[5:]
                    model_best_params[new_key] = model_best_params[key]
                    del model_best_params[key]

                model = DecisionTreeClassifier(random_state=random_state,
                                               **model_best_params)

            elif model_name == 'Random Forest Classifier':

                param_grid = [{
                    'clf':
                        [RandomForestClassifier(random_state=random_state)],
                    'clf__n_estimators':
                        [32, 64, 96, 128, ],
                    'clf__criterion':
                        ['gini', 'entropy'],
                    'clf__max_depth':
                        [8, 12, 16, 20, 24, ],
                    'clf__min_samples_leaf':
                        [1, 2, 4, 8, 16, 32, ],
                    'clf__max_features':
                        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ],
                }, ]

                grid_search = GridSearchCV(
                    pipeline,
                    param_grid=param_grid,
                    cv=5,
                    n_jobs=-1,
                    scoring='accuracy')

                grid_search.fit(x_train_, y_train)
                model_best_params = grid_search.best_params_.copy()
                model_best_params.pop('clf')
                # print(model_best_params)

                for key in list(model_best_params):
                    new_key = key[5:]
                    model_best_params[new_key] = model_best_params[key]
                    del model_best_params[key]

                model = RandomForestClassifier(random_state=random_state,
                                               **model_best_params)
            else:
                model = None

            model.fit(x_train_, y_train)
            accuracy = model.score(x_test_, y_test) * 100

            scores = decomp.inverse_transform(
                model.feature_importances_.reshape(1, -1)).reshape(-1)
            scores = np.abs(scores)
            indices = np.argsort(scores)[::-1]

            plt.subplot(len(decompositions),
                        len(model_names),
                        decomp_idx * 2 + model_idx + 1)

            plt.title('Tree Scores with %s\non %s (acc=%.1f%%)' %
                      (model_name, decomp_name, accuracy))
            plt.bar(range(x.shape[1]),
                    scores[indices],
                    edgecolor='k',
                    color='w',
                    align='center')
            plt.xticks(range(x.shape[1]), indices)
            plt.xlim([-1, x.shape[1]])

    plt.savefig('./img/indep_tree_score.png')
    if show_img:
        plt.show()
