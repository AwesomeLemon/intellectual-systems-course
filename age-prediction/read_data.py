import csv
import pandas as pd
import gc
from scipy.sparse import coo_matrix, csr_matrix

import numpy as np
import os

from util import store_obj, retrieve_obj

data_path = 'data/'
graph_path = os.path.join(data_path, 'graph')
demography_path = os.path.join(data_path, 'trainDemography')
test_users_path = os.path.join(data_path, 'users')
results_path = 'results'
extracted_features_train_path = os.path.join(data_path, 'user_features_train')
extracted_features_test_path = os.path.join(data_path, 'user_features_test')

max_id = 47289241 + 1

def load_birth_dates():
    birth_dates_path = 'data/birthdays.npy'
    if os.path.isfile(birth_dates_path):
        return np.load(birth_dates_path)

    birth_dates = np.zeros(max_id, dtype=np.int32)
    for file in [f for f in os.listdir(demography_path) if f.startswith('part')]:
        for line in csv.reader(open(os.path.join(demography_path, file)), delimiter='\t'):
            user = int(line[0])
            birth_dates[user] = int(line[2]) if line[2] != '' else 0
    np.save(birth_dates_path, birth_dates)
    return birth_dates

def load_test_users():
    test_users = []
    for line in csv.reader(open(test_users_path)):
        test_users.append(int(line[0]))
    return set(test_users)

def load_friends_birthdays(birth_dates, test_users):
    # friends_birth_dates_path = 'data/friends_birthdays.npy'
    # if os.path.isfile(friends_birth_dates_path):
    #     return np.load(friends_birth_dates_path)
    # users_to_friends_bds = np.empty(max_id, dtype=np.object)
    # users_to_friends_bds = np.empty(max_id, dtype=np.object)
    # for file in [f for f in os.listdir(graph_path) if f.startswith('part')]:
    #     for line in csv.reader(open(os.path.join(graph_path, file)),
    #                            delimiter='\t'):
    #         user = int(line[0])
    #         # if user in test_users:
    #         friends_bds = []
    #         for friendship in line[1][2:len(line[1]) - 2].split('),('):
    #             parts = friendship.split(',')
    #             friend = int(parts[0])
    #             friend_bd = birth_dates[friend]
    #             friends_bds.append(friend_bd)
    #         users_to_friends_bds[user] = friends_bds
    # np.save(friends_birth_dates_path, users_to_friends_bds)

    def save_sparse_csr(filename, array):
        np.savez(filename, data=array.data, indices=array.indices,
                 indptr=array.indptr, shape=array.shape)

    def load_sparse_csr(filename):
        loader = np.load(filename)
        return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                          shape=loader['shape'])

    csr_path = 'data/csr.npz'
    all_users_path = 'data/all_users'
    if os.path.isfile(csr_path) and os.path.isfile(all_users_path):
        print('loading')
        return load_sparse_csr(csr_path), set(retrieve_obj(all_users_path))

    links_num = 300292724
    from_user = np.zeros(links_num, dtype=np.int32)
    to_user = np.zeros(links_num, dtype=np.int32)
    data = np.zeros(links_num, dtype=np.uint16)
    current_pos = 0
    all_users = set()
    i = 0
    for file in [f for f in os.listdir(graph_path) if f.startswith('part')]:
        with open(os.path.join(graph_path, file)) as b:
            for line in csv.reader(b,
                                   delimiter='\t'):
                user = int(line[0])
                all_users.add(user)
                for friendship in line[1][2:len(line[1]) - 2].split('),('):
                    parts = friendship.split(',')
                    friend = int(parts[0])

                    from_user[current_pos] = user
                    to_user[current_pos] = friend
                    all_users.add(friend)

                    friend_bd = birth_dates[friend]
                    data[current_pos] = friend_bd #((friend_bd + 36500) << 21) + int(parts[1])

                    current_pos += 1
                i += 1
                if current_pos % 10000 == 0:
                    print(current_pos)
                    gc.collect()
    # print(current_pos)
    csr = coo_matrix((data, (from_user, to_user)), shape=(max_id, max_id)).tocsr()
    gc.collect()
    save_sparse_csr(csr_path, csr)
    store_obj(all_users, all_users_path)
    return csr, all_users

def bin(s):
    return str(s) if s <= 1 else bin(s >> 1) + str(s & 1)

def store_features(user_friends_data_matrix, users, birth_dates, path):
    with open(path, 'w+') as f:
        i = 0
        for user in users:
            friends_features = user_friends_data_matrix[user, :].data
            friends_bds = friends_features#[(friend_and_mask >> 21) - 36500 for friend_and_mask in friends_features]
            cnt_bds = len(friends_bds)
            if cnt_bds == 0:
                continue
            # del friends_features
            friends_bds.sort()
            # [friend_and_mask & 4194303 for friend_and_mask in friends_features]
            sum_bds = sum(friends_bds)
            if cnt_bds > 5:
                sum_middle = sum_bds - friends_bds[0] - friends_bds[1] - friends_bds[-1] - friends_bds[-2]
            else:
                sum_middle = sum_bds
            f.write(str(user) + ',' + str(sum_bds) + ',' + str(sum_middle) + ',' + str(cnt_bds) + ',' + str(birth_dates[user]) + '\n')
            i += 1
            if i % 10000 == 0:
                print(i)
                gc.collect()

def load_train_features():
    return np.array(pd.read_csv(extracted_features_train_path, delimiter=",", header=None).as_matrix())

def load_test_features():
    csv = pd.read_csv(extracted_features_test_path, delimiter=",", header=None)
    return np.array(csv.as_matrix())

if __name__ == '__main__':
    test_users = load_test_users()
    birth_dates = load_birth_dates()
    friends_bds, all_users = load_friends_birthdays(birth_dates, test_users)
    train_users = all_users - test_users
    # store_features(friends_bds, train_users, birth_dates, extracted_features_train_path)
    # store_features(friends_bds, test_users, birth_dates, extracted_features_test_path)

    #predicting via average:
    # with open('pred.txt', 'w') as f:
    #     for user, user_friends in enumerate(friends_bds):
    #         if user_friends is None:
    #             continue
    #         user_friends.sort()
    #         if len(user_friends) > 5:
    #             # user_friends = user_friends[int(len(user_friends) * .05): int(len(user_friends) * .95)]
    #             user_friends = user_friends[2: len(user_friends) - 2]
    #         predicted_age = sum(user_friends) / len(user_friends)
    #         # predicted_age = user_friends[len(user_friends) / 2]
    #         f.write(str(user) + ',' + str(predicted_age) + '\n')

    #enriching features via average
    with open('pred.txt', 'w') as f:
        for user, user_friends in enumerate(friends_bds):
            if user_friends is None:
                continue
            user_friends.sort()
            if len(user_friends) > 5:
                # user_friends = user_friends[int(len(user_friends) * .05): int(len(user_friends) * .95)]
                user_friends = user_friends[2: len(user_friends) - 2]
            predicted_age = sum(user_friends) / len(user_friends)
            # predicted_age = user_friends[len(user_friends) / 2]
            f.write(str(user) + ',' + str(predicted_age) + '\n')