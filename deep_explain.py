""" 
    File Name:          deep_explain.py 
    Project Name:       FeatureImportance
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               6/25/2018
    Python Version:     3.6.4
    File Description:   

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras import backend as K
from deepexplain.tensorflow import DeepExplain


def deep_explain(data: dict,
                 target: np.ndarray,
                 num_features: int,
                 random_state: int = 0,
                 show_img: bool = False):

    # Feature ranking with DeepExplain
    # https://github.com/marcoancona/DeepExplain

    np.random.seed(random_state)
    x = data['Normalized Data']
    x_train, x_test, y_train, y_test = \
        train_test_split(x, target, random_state=random_state)

    if num_features != data['Normalized Data'].shape[1]:

        print('Performing PCA ...')
        pca = PCA(n_components=num_features)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

    model = Sequential()
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=256,
              epochs=500,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', score[1])

    with DeepExplain(session=K.get_session()) as de:

        input_tensor = model.layers[0].input
        tmp = Model(inputs=input_tensor, outputs=model.layers[-2].output)
        target_tensor = tmp(input_tensor)
        print(target_tensor)
        print(y_test.shape)

        attributions = {
            'Saliency Maps':
                de.explain('saliency', target_tensor, input_tensor, x_test),
            'Gradient * Input':
                de.explain('grad*input', target_tensor, input_tensor, x_test),
            'Integrated Gradients':
                de.explain('intgrad', target_tensor, input_tensor, x_test),
            'Epsilon-LRP':
                de.explain('elrp', target_tensor, input_tensor, x_test),
            'DeepLIFT':
                de.explain('deeplift', target_tensor, input_tensor, x_test),
            'Occlusion':
                de.explain('occlusion', target_tensor, input_tensor, x_test),
        }

    plt.figure(figsize=(16, 6 * len(attributions)))

    for method_idx, (method_name, attr) in enumerate(attributions.items()):

        summed_attr = np.sum(attr, axis=0)
        summed_abs_attr = np.sum(np.abs(attr), axis=0)

        for data_idx, (scores) in enumerate([summed_attr, summed_abs_attr]):

            if num_features != data['Normalized Data'].shape[1]:
                scores = pca.inverse_transform(scores)
                scores = np.abs(scores)

            indices = np.argsort(scores)[::-1]
            plt.subplot(len(attributions), 2, method_idx * 2 + data_idx + 1)

            if data_idx == 0:
                plt.title('Feature Attribution using %s' % method_name)
            else:
                plt.title('Feature Attribution using Abs(%s)' % method_name)

            plt.bar(range(x.shape[1]),
                    scores[indices],
                    edgecolor='k',
                    color='w',
                    align='center')
            plt.xticks(range(x.shape[1]), indices)
            plt.xlim([-1, x.shape[1]])

    if num_features != data['Normalized Data'].shape[1]:
        plt.savefig('./img/pca(%d)_deep_explain.png' % num_features)
    else:
        plt.savefig('./img/deep_explain.png')

    if show_img:
        plt.show()





