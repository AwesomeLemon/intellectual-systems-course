triplets_total = 48373586 / 10
occurence_threshold = 420


def prepare_data(filepath):
    u_exist = {}
    i_exist = {}

    def line_to_triplet(line):
        vals = line.strip().split('\t')
        u = vals[0]
        i = vals[1]
        play_count = int(vals[2])
        return i, play_count, u

    def find_users_items_above_threshold(f, threshold, samples_to_read):
        play_counts = {}
        user_count = {}
        item_count = {}

        for cnt in range(samples_to_read):
            line = f.readline()
            if line == '':
                break
            i, play_count, u = line_to_triplet(line)

            if play_count not in play_counts:
                play_counts[play_count] = 0
            play_counts[play_count] += 1

            i_index = index_object(i, i_exist)
            u_index = index_object(u, u_exist)

            if i_index not in item_count:
                item_count[i_index] = 0
            item_count[i_index] += 1

            if u_index not in user_count:
                user_count[u_index] = 0
            user_count[u_index] += 1

            if cnt % 200000 == 0:
                print cnt

        good_users = set([user for (user, count) in user_count.iteritems() if count > threshold])
        good_items = set([item for (item, count) in item_count.iteritems() if count > threshold])
        return good_users, good_items

    def index_object(obj, obj_exist):
        if obj not in obj_exist:
            index = len(obj_exist)
            obj_exist[obj] = index
        else:
            index = obj_exist[obj]
        return index

    def read_triplets(f, lines_total, good_users, good_items):
        def convert_play_count_to_rating(play_count):
            if play_count == 1:
                return 1
            if play_count < 6:
                return 2
            return 3

        triplets = {}

        rating_sum = 0
        for cnt in range(lines_total):
            line = f.readline()
            if line == '':
                break
            i, play_count, u = line_to_triplet(line)

            i_index = index_object(i, i_exist)
            if i_index not in good_items:
                continue

            u_index = index_object(u, u_exist)
            if u_index not in good_users:
                continue

            rating = convert_play_count_to_rating(play_count)
            triplets[(i_index, u_index)] = rating
            rating_sum += rating
        return triplets, rating_sum

    with open(filepath) as f:
        # if not os.path.isfile(stored_users_file) or not os.path.isfile(stored_items_file):
        good_users, good_items = find_users_items_above_threshold(f, occurence_threshold, triplets_total)
        # else:
        #     good_users = util.retrieve_obj(stored_users_file)
        #     good_items = util.retrieve_obj(stored_items_file)

        print 'good users# = ' + str(len(good_users))
        print 'good items# = ' + str(len(good_items))
    with open(filepath) as f:
        triplets, rating_sum = read_triplets(f, triplets_total, good_users, good_items)

    return triplets, rating_sum / float(triplets_total), good_items
