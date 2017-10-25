import numpy as np
from metrics import mae, rmse, ndcg
from read_data import prepare_data


def cross_validation(all_data_dict, avg, items, part_num=5):
    def separate_in_parts(all_triplets_dict, part_num):
        all_data_list = all_triplets_dict.items()
        np.random.shuffle(all_data_list)
        total = len(all_data_list)
        part_len = total / float(part_num)
        parts = []
        for i in range(part_num):
            parts.append(all_data_list[int(i * part_len): int((i + 1) * part_len)])
        return parts

    def triplets_list_to_dict(triplets_list):
        res = {}
        for triplet in triplets_list:
            res[(triplet[0][0], triplet[0][1])] = triplet[1]

        return res

    parts = separate_in_parts(all_data_dict, part_num)
    mae_sum = 0.0
    rmse_sum = 0.0
    ndcg_sum = 0.0
    for i in range(part_num):
        test = triplets_list_to_dict(parts[i])
        train = []
        for j in range(part_num):
            if i == j:
                continue
            train += parts[j]
        train = triplets_list_to_dict(train)

        mae, rmse, ndcg = sgd(train, test, avg, items)
        print '-' * 10
        print 'cross-validation No ' + str(i)
        print 'mae = ' + str(mae) \
              + ', rmse = ' + str(rmse) \
              + ', ndcg = ' + str(ndcg)
        print '-' * 10

        mae_sum += mae
        rmse_sum += rmse
        ndcg_sum += ndcg

    return mae_sum / part_num, rmse_sum / part_num, ndcg_sum / part_num


def sgd(train, test, avg, all_items, learning_rate=0.001, lambda4=0.01, hidden_dim=70, epochs=50):
    print 'gamma = ' + str(learning_rate)
    print 'lambda4 = ' + str(lambda4)
    print 'hidden_dim = ' + str(hidden_dim)
    print 'train = ' + str(len(train))
    print 'test = ' + str(len(test))

    lr_decay = 1.0
    print 'decay = ' + str(lr_decay)
    print 'mae_test\t:\trmse_test\t:\tmae_train\t:\trmse_train'

    i_total = max(all_items) + 1
    u_total = max(max(zip(*train.keys())[1]),
                  max(zip(*test.keys())[1])) + 1  # I reaaally don't want to reenumerate everything

    q = np.random.rand(i_total, hidden_dim)  # .astype('float64')
    p = np.random.rand(u_total, hidden_dim)  # .astype('float64')
    # q = np.full((i_total, hidden_dim), 0.1)
    # p = np.full((u_total, hidden_dim), 0.1)
    user_bias = np.random.rand(u_total, 1)
    item_bias = np.random.rand(i_total, 1)
    test_users = zip(*test.keys())[1]

    get_approx_rating = lambda u, i: avg + user_bias[u][0] + item_bias[i][0] + q[i].dot(p[u])
    get_true_rating_test = lambda u, i: test[(i, u)] if (i, u) in test else None

    for epoch in range(epochs):
        for key, value in train.iteritems():
            i = key[0]
            u = key[1]

            r_iu = value
            r_predict = get_approx_rating(u, i)
            e = r_iu - r_predict

            q_gradient = e * p[u] - lambda4 * q[i]
            q_gradient /= np.linalg.norm(q_gradient)
            q[i] += learning_rate * (q_gradient)

            p_gradient = e * q[i] - lambda4 * p[u]
            p_gradient /= np.linalg.norm(p_gradient)
            p[u] += learning_rate * (p_gradient)

            user_bias[u] += learning_rate * (e - lambda4 * user_bias[u])
            item_bias[i] += learning_rate * (e - lambda4 * item_bias[i])

        learning_rate *= lr_decay

        if epoch % 5 == 0 or epoch + 1 == epochs:
            mae_cur = mae(test, get_approx_rating)
            rmse_cur = rmse(test, get_approx_rating)

            print str(epoch) + ": " \
                  + str(mae_cur) + " : " \
                  + str(rmse_cur) + " : " \
                  + str(mae(train, get_approx_rating)) + " : " \
                  + str(rmse(train, get_approx_rating))
            #+ str(ndcg_cur) + " : " \
    ndcg_cur = ndcg(test_users, all_items, get_true_rating_test, get_approx_rating)
    return mae_cur, rmse_cur, ndcg_cur


data_dict, avg, items = prepare_data('/home/alex/Downloads/train_triplets.txt')
mae_avg, rmse_avg, ndcg_avg = cross_validation(data_dict, avg, items)
print '-' * 10
print 'Average:'
print 'mae = ' + str(mae_avg) \
      + ', rmse = ' + str(rmse_avg) \
      + ', ndcg = ' + str(ndcg_avg)
print '-' * 10