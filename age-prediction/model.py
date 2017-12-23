from keras import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
from sklearn import linear_model

from read_data import load_train_features, load_test_users, load_test_features
import numpy as np

cur_model_name = '50_50_05_50_05_50_1epoch5mse_adagrad_div1000_yesshuffle_normalised_avgfeature.h5'
continue_training = False
y_stats_path = 'y_stats'
x_stats_path = 'x_stats'
improved_train_features_path = 'data/user_features_train_improved'
improved_test_features_path = 'data/user_features_test_improved'


def improve_train_features():
    data = load_train_features()
    np.random.shuffle(data)
    x = data[:, 1:4]
    x_new = np.zeros(shape=x.shape)
    x_stats = np.zeros((3, 2))
    for i in range(0, 3):
        cur = x[:, i]
        mean = cur.mean(axis=0)
        std = cur.std(axis=0)
        cur = (cur - mean) / std
        x_new[:, i] = cur
        x_stats[i, 0] = mean
        x_stats[i, 1] = std
    np.savetxt(x_stats_path, x_stats)

    y = data[:, 4]
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    y = (y - y_mean) / y_std
    y_stats = np.array([y_mean, y_std])
    np.savetxt(y_stats_path, y_stats)
    average_feature = x_new[:, 1] / x_new[:, 2]
    average_feature = (average_feature - average_feature.mean(axis=0)) / average_feature.std(axis=0)
    # average_feature /= 10.0
    np.savetxt(improved_train_features_path, np.c_[x_new, average_feature, y], delimiter=',')

def improve_test_features():
    data = load_test_features()
    x = data[:, 1:4]
    x_new = np.zeros(shape=x.shape)
    x_stats = np.loadtxt(x_stats_path)
    for i in range(0, 3):
        cur = x[:, i]
        x_new[:, i] = (cur - x_stats[i, 0]) / x_stats[i, 1]
    average_feature = x_new[:, 1] / x_new[:, 2]
    average_feature = (average_feature - average_feature.mean(axis=0)) / average_feature.std(axis=0)
    # average_feature /= 10.0
    np.savetxt(improved_test_features_path, np.c_[data[:, 0], x_new, average_feature], delimiter=',')


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
    y_stats = np.loadtxt('y_stats')
    with open('nn_pred.txt', 'w+') as f:
        user_ages = zip(test_users, result)
        user_ages.sort(key=lambda tup: tup[0])
        for user, age in user_ages:
            f.write(str(user) + ',' + str(int(age[0] * y_stats[1] + y_stats[0])) + '\n')


# train()
# predict()
# improve_train_features()
# improve_test_features()
lin_reg()