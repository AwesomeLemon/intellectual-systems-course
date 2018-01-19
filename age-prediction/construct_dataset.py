from read_data import load_train_features, load_test_users, load_test_features
import numpy as np

y_stats_path = 'y_stats'
x_stats_path = 'x_stats'
normalised_train_features_path = 'data/user_features_train_improved'
normalised_test_features_path = 'data/user_features_test_improved'

average_train_features_path = 'data/avg_features_train'
average_test_features_path = 'data/avg_features_test'

def construct_averages_dataset(is_train=True):
    if is_train:
        data = load_train_features()
    else:
        data = load_test_features()
    x = data[:, 1:5]
    x_averages = np.zeros(shape=(x.shape[0], 2))

    x_averages[:, 0] = x[:, 0] / x[:, 2]
    x_averages[:, 1] = x[:, 1] / x[:, 2]
    x_averages /= 1000.0

    if is_train:
        y = data[:, 5]
        y = np.divide(y, 1000.0)
        np.savetxt(average_train_features_path, np.c_[x_averages, y], delimiter=',')
    else:
        np.savetxt(average_test_features_path, np.c_[data[:, 0], x_averages], delimiter=',')

def normalise_train():
    data = load_train_features()
    # np.random.shuffle(data)
    x = data[:, 1:5]
    x_new = np.zeros(shape=x.shape)
    x_stats = np.zeros((4, 2))

    for i in range(0, 4):
        x_new[:, i], x_stats[i, 0], x_stats[i, 1] = normalise(x[:, i])

    np.savetxt(x_stats_path, x_stats)

    y = data[:, 5]
    y, y_mean, y_std = normalise(y)
    y_stats = np.array([y_mean, y_std])
    np.savetxt(y_stats_path, y_stats)

    average_feature = x_new[:, 1] / x_new[:, 2]
    average_feature = (average_feature - average_feature.mean(axis=0)) / average_feature.std(axis=0)
    # average_feature /= 10.0
    np.savetxt(normalised_train_features_path, np.c_[x_new, average_feature, y], delimiter=',')


def normalise(one_feature_array):
    mean = one_feature_array.mean(axis=0)
    std = one_feature_array.std(axis=0)
    one_feature_array = (one_feature_array - mean) / std
    return one_feature_array, mean, std


def normalise_test():
    data = load_test_features()
    x = data[:, 1:5]
    x_new = np.zeros(shape=x.shape)
    x_stats = np.loadtxt(x_stats_path)
    for i in range(0, 4):
        x_new[:, i] = (x[:, i] - x_stats[i, 0]) / x_stats[i, 1]
    average_feature = x_new[:, 1] / x_new[:, 2]
    average_feature = (average_feature - average_feature.mean(axis=0)) / average_feature.std(axis=0)
    # average_feature /= 10.0
    np.savetxt(normalised_test_features_path, np.c_[data[:, 0], x_new, average_feature], delimiter=',')

# construct_averages_dataset(False)

normalise_train()
normalise_test()