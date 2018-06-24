""" 
    File Name:          tree_score.py 
    Project Name:       FeatureImportance
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               6/24/2018
    Python Version:     3.6.4
    File Description:   

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def tree_score(data: dict,
               target: np.ndarray,
               params: dict,
               models: dict = None,
               random_state: int = 0,
               show_img: bool = False):

    # Feature importance using tree-based scoring on :
    #     different classifiers;
    #     different hyper-parameters (default vs optimal);
    #     different random seeds;

    if models is None:
        models = {
            'Decision Tree Classifier '
            '(default config, random=%d)' % random_state:
                DecisionTreeClassifier(
                    random_state=random_state),

            'Decision Tree Classifier '
            '(best config, random=%d)' % random_state:
                DecisionTreeClassifier(
                    random_state=random_state,
                    **params['Decision Tree Classifier on Original Data']),

            'Decision Tree Classifier '
            '(best config, random=%d)' % (random_state + 1):
                DecisionTreeClassifier(
                    random_state=(random_state + 1),
                    **params['Decision Tree Classifier on Original Data']),

            'Random Forest Classifier '
            '(default config, random=%d)' % random_state:
                RandomForestClassifier(
                    random_state=random_state),

            'Random Forest Classifier '
            '(best config, random=%d)' % random_state:
                RandomForestClassifier(
                    random_state=random_state,
                    **params['Random Forest Classifier on Original Data']),

            'Random Forest Classifier '
            '(best config, random=%d)' % (random_state + 1):
                RandomForestClassifier(
                    random_state=(random_state + 1),
                    **params['Random Forest Classifier on Original Data']), }

    plt.figure(figsize=(8 * len(data), 6 * len(models)))

    for model_idx, (model_name, model) in enumerate(models.items()):

        for data_idx, (data_name, x) in enumerate(data.items()):

            x_train, x_test, y_train, y_test = \
                train_test_split(x, target, random_state=random_state)

            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test) * 100

            scores = model.feature_importances_
            indices = np.argsort(scores)[::-1]

            plt.subplot(len(models), len(data), model_idx * 2 + data_idx + 1)
            plt.title('Tree Scores using %s\non %s (acc=%.1f%%)' %
                      (model_name, data_name, accuracy))
            plt.bar(range(x.shape[1]),
                    scores[indices],
                    edgecolor='k',
                    color='w',
                    align='center')
            plt.xticks(range(x.shape[1]), indices)
            plt.xlim([-1, x.shape[1]])

    plt.savefig('./img/tree_score.png')
    if show_img:
        plt.show()
