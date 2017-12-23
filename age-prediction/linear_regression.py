from keras import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
from sklearn import linear_model

from read_data import load_train_features, load_test_users, load_test_features
import numpy as np

improved_train_features_path = 'data/user_features_train_improved'
improved_test_features_path = 'data/user_features_test_improved'

def lin_reg():
    data = np.loadtxt(improved_train_features_path, delimiter=",")
    x = data[:, :4]
    y = data[:, 4]
    train_test_split = int(0.9 * len(x))
    x_train, x_test = x[:train_test_split], x[train_test_split:]
    y_train, y_test = y[:train_test_split], y[train_test_split:]
    clf = linear_model.LinearRegression()
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))

    test_users = load_test_users()
    test_users = list(test_users)
    test_users.sort()
    test_features = np.loadtxt(improved_test_features_path, delimiter=',')
    # test_features[:, 0] = int(test_features[:, 0])
    test_features = test_features[test_features[:,0].argsort()]
    # test_features = test_features[:, 1:]
    y_stats = np.loadtxt('y_stats')
    with open('nn_pred.txt', 'w+') as f:
        for user in test_users:
            test_feat_ind = 1
            while test_features[test_feat_ind-1, 0] < user and (test_feat_ind - 1) * 2 < len(test_features):
                test_feat_ind *= 2
            while test_features[test_feat_ind-1, 0] != user:
                test_feat_ind -= 1
            ind_ = test_features[test_feat_ind, 1:]
            if test_features[test_feat_ind - 1, 0] != user:
                print('fuck!')
            predict = clf.predict(ind_)
            f.write(str(user) + ',' + str(int(predict * y_stats[1] + y_stats[0])) + '\n')
            print(user)