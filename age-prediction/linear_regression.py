from sklearn import linear_model

from read_data import load_train_features, load_test_users, load_test_features
import numpy as np
import construct_dataset

def lin_reg(is_normalised=True):
    is_averaged = not is_normalised
    if is_normalised:
       data = np.loadtxt(construct_dataset.normalised_train_features_path, delimiter=",")
       x = data[:, :4]
       y = data[:, 4]
    else:
        data = np.loadtxt(construct_dataset.average_train_features_path, delimiter=",")
        x = data[:, :2]
        y = data[:, 2]
    train_test_split = int(0.9 * len(x))
    x_train, x_test = x[:train_test_split], x[train_test_split:]
    y_train, y_test = y[:train_test_split], y[train_test_split:]
    clf = linear_model.LinearRegression()
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))

    test_users = load_test_users()
    test_users = list(test_users)
    test_users.sort()
    if is_normalised:
        test_features = np.loadtxt(construct_dataset.normalised_test_features_path, delimiter=',')
    else:
        test_features = np.loadtxt(construct_dataset.average_test_features_path, delimiter=',')
    # test_features[:, 0] = int(test_features[:, 0])
    test_features = test_features[test_features[:, 0].argsort()]
    # test_features = test_features[:, 1:]
    if is_normalised:
        y_stats = np.loadtxt('y_stats')
    with open('nn_pred.txt', 'w+') as f:
        for i, user in enumerate(test_users):
            user_features = test_features[i, 1:]
            predict = clf.predict(user_features)
            if is_normalised:
                f.write(str(user) + ',' + str(int(predict * y_stats[1] + y_stats[0])) + '\n')
            else:
                f.write(str(user) + ',' + str(int(predict[0] * 1000.0)) + '\n')
            print(user)



lin_reg(False)