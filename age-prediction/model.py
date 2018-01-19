from keras import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
from sklearn import linear_model

from read_data import load_train_features, load_test_users, load_test_features
import numpy as np

import construct_dataset

y_stats_path = 'y_stats'
x_stats_path = 'x_stats'
cur_model_name = '50_50_50_50_1epoch5mse_adagrad_div1000_additionalaveragefeature_normalised.h5'
continue_training = False
is_normalised=True
is_averaged = not is_normalised


def train():
    # data = load_train_features()
    # np.random.shuffle(data)
    # x = data[:, 1:4]
    # x[:, 0] = x[:, 0] / 1000.0
    # x[:, 1] = x[:, 1] / 1000.0
    # y = data[:, 4] / 1000.0
    if is_normalised:
       data = np.loadtxt(construct_dataset.normalised_train_features_path, delimiter=",")
       x = data[:, :5]
       y = data[:, 5]
    else:
        data = np.loadtxt(construct_dataset.average_train_features_path, delimiter=",")
        x = data[:, :2]
        y = data[:, 2]
    train_test_split = int(0.9 * len(x))
    x_train, x_test = x[:train_test_split], x[train_test_split:]
    y_train, y_test = y[:train_test_split], y[train_test_split:]
    if continue_training:
        model = load_model(cur_model_name)
    else:
        model = Sequential()
        model.add(Dense(50, input_dim=5, activation='relu'))
        model.add(Dense(50, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(50, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adagrad')
    # Fit the model
    model.fit(x_train, y_train, epochs=5, batch_size=10, verbose=1, validation_data=(x_test, y_test))
    # calculate predictions
    model.save(cur_model_name)



def predict():
    test_users = load_test_users()
    # test_features = load_test_features()
    # test_features = test_features[:, 1:]
    # test_features[:, :2] = test_features[:, :2] / 1000.0
    if is_normalised:
        test_features = np.loadtxt(construct_dataset.normalised_test_features_path, delimiter=',')
    else:
        test_features = np.loadtxt(construct_dataset.average_test_features_path, delimiter=',')
    test_features = test_features[:, 1:]
    model = load_model(cur_model_name)
    result = model.predict(test_features)
    y_stats = np.loadtxt(y_stats_path)
    with open('nn_pred.txt', 'w+') as f:
        user_ages = zip(test_users, result)
        user_ages.sort(key=lambda tup: tup[0])
        for user, age in user_ages:
            if is_normalised:
                f.write(str(user) + ',' + str(int(age[0] * y_stats[1] + y_stats[0])) + '\n')
            else:
                f.write(str(user) + ',' + str(int(age[0] * 1000.0)) + '\n')


# train()
predict()
# improve_train_features()
# improve_test_features()
# lin_reg()