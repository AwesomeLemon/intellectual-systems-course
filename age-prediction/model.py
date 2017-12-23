from keras import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
from sklearn import linear_model

from read_data import load_train_features, load_test_users, load_test_features
import numpy as np

y_stats_path = 'y_stats'
x_stats_path = 'x_stats'
cur_model_name = '50_50_05_50_05_50_1epoch5mse_adagrad_div1000_yesshuffle_normalised_avgfeature.h5'
continue_training = False
improved_train_features_path = 'data/user_features_train_improved'
improved_test_features_path = 'data/user_features_test_improved'


def train():
    # data = load_train_features()
    # np.random.shuffle(data)
    # x = data[:, 1:4]
    # x[:, 0] = x[:, 0] / 1000.0
    # x[:, 1] = x[:, 1] / 1000.0
    # y = data[:, 4] / 1000.0
    data = np.loadtxt(improved_train_features_path, delimiter=",")
    x = data[:, :4]
    y = data[:, 4]
    train_test_split = int(0.9 * len(x))
    x_train, x_test = x[:train_test_split], x[train_test_split:]
    y_train, y_test = y[:train_test_split], y[train_test_split:]
    if continue_training:
        model = load_model(cur_model_name)
    else:
        model = Sequential()
        model.add(Dense(50, input_dim=4, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.5))
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
    test_features = np.loadtxt(improved_test_features_path, delimiter=',')
    test_features = test_features[:, 1:]
    model = load_model(cur_model_name)
    result = model.predict(test_features)
    y_stats = np.loadtxt(y_stats_path)
    with open('nn_pred.txt', 'w+') as f:
        user_ages = zip(test_users, result)
        user_ages.sort(key=lambda tup: tup[0])
        for user, age in user_ages:
            f.write(str(user) + ',' + str(int(age[0] * y_stats[1] + y_stats[0])) + '\n')


# train()
# predict()
# improve_train_features()
# improve_test_features()
# lin_reg()