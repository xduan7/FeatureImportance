""" 
    File Name:          get_data.py 
    Project Name:       FeatureImportance
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               6/24/2018
    Python Version:     3.6.4
    File Description:   
        Generate data for feature ranking purpose
"""

import pickle
import numpy as np
import os.path
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

RAND_SEED = 0


def get_data(num_samples: int = 5000,
             num_features: int = 20,
             num_redundant: int = 2,
             num_repeated: int = 2,
             num_informative: int = 8,
             summary: bool = True,
             show_img: bool = False,
             random_state: int = RAND_SEED):

    file_name = './data/data_%d.dat' % random_state
    if os.path.isfile(file_name):
        file = open(file_name, 'rb')
        (original_x, normalized_x, y) = pickle.load(file)
        print('Using existing data ...')
    else:
        original_x, y = make_classification(
            n_samples=num_samples,
            n_features=num_features,
            n_informative=num_informative,
            n_redundant=num_redundant,
            n_repeated=num_repeated,
            n_classes=2,
            flip_y=0.1,
            shift=None,
            scale=None,
            random_state=random_state,
            shuffle=False)

        scaler = StandardScaler()
        normalized_x = scaler.fit_transform(original_x)

    if summary:

        # Plot the feature mean and variances of all features
        plt.figure(figsize=(16, 12))
        indices = range(num_features)

        plt.subplot(2, 2, 1)
        plt.title("Feature Mean for Original Data")
        plt.bar(range(original_x.shape[1]),
                np.mean(original_x, axis=0)[indices],
                edgecolor='k',
                color='w',
                align='center')
        plt.xticks(range(original_x.shape[1]), indices)
        plt.xlim([-1, original_x.shape[1]])

        plt.subplot(2, 2, 2)
        axes = plt.gca()
        axes.set_ylim([-1e-13, 1e-13])
        plt.title("Feature Mean for Normalized Data")
        plt.bar(range(normalized_x.shape[1]),
                np.mean(normalized_x, axis=0)[indices],
                edgecolor='k',
                color='w',
                align='center')
        plt.xticks(range(normalized_x.shape[1]), indices)
        plt.xlim([-1, normalized_x.shape[1]])

        plt.subplot(2, 2, 3)
        plt.title("Feature Variances for Original Data")
        plt.bar(range(original_x.shape[1]),
                np.var(original_x, axis=0)[indices],
                edgecolor='k',
                color='w',
                align='center')
        plt.xticks(range(original_x.shape[1]), indices)
        plt.xlim([-1, original_x.shape[1]])

        plt.subplot(2, 2, 4)
        plt.title("Feature Variances for Normalized Data")
        plt.bar(range(normalized_x.shape[1]),
                np.var(normalized_x, axis=0)[indices],
                edgecolor='k',
                color='w',
                align='center')
        plt.xticks(range(normalized_x.shape[1]), indices)
        plt.xlim([-1, normalized_x.shape[1]])

        plt.savefig('./img/data_summary.png')
        if show_img:
            plt.show()

    file = open(file_name, 'wb')
    pickle.dump((original_x, normalized_x, y), file)

    return original_x, normalized_x, y
