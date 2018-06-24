""" 
    File Name:          grid_search.py 
    Project Name:       FeatureImportance
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               6/24/2018
    Python Version:     3.6.4
    File Description:   

"""

import pickle
import numpy as np
import os.path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import \
    ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier


# Parameter list for grid search
n_estimators_list = [8, 16, 32, 64, ]
criterion_list = ['gini', 'entropy']
max_depth_list = [4, 8, 12, 16, 20, 24, 28, 32, ]
min_samples_leaf_list = [1, 2, 3, 4, 5, 6, 7, 8, ]
max_features_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ]
learning_rate_list = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, ]
base_estimator_list = [
    DecisionTreeClassifier(criterion='entropy', max_depth=2),
    DecisionTreeClassifier(criterion='entropy', max_depth=4),
    DecisionTreeClassifier(criterion='entropy', max_depth=6),
    DecisionTreeClassifier(criterion='entropy', max_depth=8), ]


def grid_search(data: dict,
                target: np.ndarray,
                random_state: int,
                model_param_grid: dict = None):

    file_name = './data/best_params_%d.dat' % random_state
    if os.path.isfile(file_name):
        file = open(file_name, 'rb')
        print('Using existing parameters ...')
        return pickle.load(file)

    if model_param_grid is None:
        model_param_grid = {
            # Parameter grid for decision tree
            'Decision Tree Classifier': [{
                'clf': [DecisionTreeClassifier(random_state=random_state)],
                'clf__criterion': criterion_list,
                'clf__max_depth': max_depth_list,
                'clf__min_samples_leaf': min_samples_leaf_list,
                'clf__max_features': max_features_list,
            }, ],

            # # Parameter grid for extra tree
            # 'Extra Tree Classifier': [{
            #     'clf': [ExtraTreesClassifier(random_state=random_state)],
            #     'clf__n_estimators': n_estimators_list,
            #     'clf__criterion': criterion_list,
            #     'clf__max_depth': max_depth_list,
            #     'clf__min_samples_leaf': min_samples_leaf_list,
            #     'clf__max_features': max_features_list,
            # }, ],

            # # Parameter grid for adaboost
            # 'AdaBoost Classifier': [{
            #     'clf': [AdaBoostClassifier(random_state=random_state)],
            #     'clf__base_estimator': base_estimator_list,
            #     'clf__n_estimators': n_estimators_list,
            #     'clf__learning_rate': learning_rate_list,
            # }, ],

            # Parameter grid for random forest
            'Random Forest Classifier': [{
                'clf': [RandomForestClassifier(random_state=random_state)],
                'clf__n_estimators': n_estimators_list,
                'clf__criterion': criterion_list,
                'clf__max_depth': max_depth_list,
                'clf__min_samples_leaf': min_samples_leaf_list,
                'clf__max_features': max_features_list,
            }, ],

            # Parameter grid for knn
            'Nearest Neighbor Classifier': [{
                'clf': [KNeighborsClassifier(algorithm='brute')],
                'clf__n_neighbors': range(5, 129, 4),
            }, ],

            # Parameter grid for linear SVM
            'Linear Support Vector Classifier': [{
                'clf': [LinearSVC(dual=False,
                                  random_state=random_state,
                                  max_iter=1e5)],
                'clf__penalty': ['l1', 'l2'],
                'clf__C': [1e-4, 1e-3, 1e-2, 1e-1, ],
            }, ],

            # Parameter grid for RBF SVM
            'RBF Support Vector Classifier': [{
                'clf': [SVC(random_state=random_state, max_iter=1e5)],
                'clf__C': [1, 2, 4, 8, 16, 32, 64, ],
            }, ],
        }

    # First find the best hyper params for each model
    best_params = {}

    for model_name, param_grid in model_param_grid.items():

        print('Searching best params for %s ...' % model_name)

        for data_name, x in data.items():

            pipeline = Pipeline(steps=[('clf', None), ])
            model = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=5,
                n_jobs=-1,
                scoring='accuracy')

            model.fit(x, target)

            model_best_params = model.best_params_.copy()
            model_best_params.pop('clf')

            # Replace the keys so that parameters can be reused
            for key in list(model_best_params):
                new_key = key[5:]
                model_best_params[new_key] = model_best_params[key]
                del model_best_params[key]

            best_params[model_name + ' on ' + data_name] = model_best_params

    file = open(file_name, 'wb')
    pickle.dump(best_params, file)

    return best_params
