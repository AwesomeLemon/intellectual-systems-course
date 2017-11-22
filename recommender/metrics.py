from math import sqrt, log
from sortedcontainers import SortedList


def mae(test, approximate):
    res = 0.0
    for key, value in test.iteritems():
        res += abs(approximate(key[1], key[0]) - value)

    res /= len(test)
    return res


def rmse(test, approximate):
    res = 0.0
    for key, value in test.iteritems():
        res += pow(approximate(key[1], key[0]) - value, 2)

    res = sqrt(res / len(test))
    return res


def ndcg(users, items, get_true_rating, get_approx_rating, size=5):
    def dcg_helper(relevances, size):
        if len(relevances) < size:
            relevances += [0]*(size - len(relevances))
        dcg = 0.0
        for i, rel in enumerate(relevances):
            dcg += rel / max(1.0, log(i + 2, 2))
        return dcg

    def get_most_relevant(u_index, all_i_indices, get_rating, size):
        most_relevant = SortedList()
        least_relevant_one = -1
        for i_index in all_i_indices:
            rating = get_rating(u_index, i_index)
            if rating is None:
                continue
            if rating > least_relevant_one:
                if len(most_relevant) == size:
                    most_relevant.pop(0)
                most_relevant.add(rating)
                least_relevant_one = most_relevant[0]

        return most_relevant

    # test_rels = [3, 2, 3, 0, 1, 2]
    # test_dcg = dcg_helper(test_rels, 6)
    # test_idcg = dcg_helper(list(reversed(sorted(test_rels))), 6)
    # print test_dcg
    # print test_idcg

    ndcg_total = 0.0
    for u in users:
        true_relevant = get_most_relevant(u, items, get_true_rating, size)
        idcg = dcg_helper(true_relevant, size)

        approx_relevant = get_most_relevant(u, items, get_approx_rating, size)
        dcg = dcg_helper(approx_relevant, size)

        cur_ndcg = dcg / idcg
        ndcg_total += cur_ndcg

    ndcg_total /= len(users)
    return ndcg_total

    # ndcg(None, None, None, None)
