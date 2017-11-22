import keras
from keras.layers import Bidirectional, Embedding
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import pandas as pd
import numpy as np
import data_util
from embeddings import get_ruwiki_embedding_keras, get_ruwiki_embedding_dict

max_features = 20000
max_len = 20
batch_size = 32
epochs = 3

def train_model():
    data = get_data('data/cleaned_data.csv')
    np.random.shuffle(data)

    xs, _ = get_xs(data)
    # xs = [x.decode('utf-8') for x in xs]
    best_words_set = data_util.construct_good_set(xs, max_features, 0)
    # pretrained_embedding_vocab = get_ruwiki_embedding_dict()
    # data_util.sentences_to_predefined_scalars(xs, best_words_set, pretrained_embedding_vocab)
    data_util.sentences_to_scalars(xs, best_words_set)

    # def count_twitter_words_in_pretrained_vocab(xs_dict, pretrained_vocab):
    #     i = 0
    #     for x in xs_dict:
    #         if x in pretrained_vocab:
    #             i += 1
    #     print i
    #
    # count_twitter_words_in_pretrained_vocab(best_words_set, pretrained_embedding_vocab)


    train_test_split = int(0.8 * len(xs))
    xs_train = xs[:train_test_split]
    xs_test = xs[train_test_split:]

    xs_train_scalar = keras.preprocessing.sequence.pad_sequences(xs_train, maxlen=max_len, padding='post',
                                                                 truncating='post')
    xs_test_scalar = keras.preprocessing.sequence.pad_sequences(xs_test, maxlen=max_len, padding='post',
                                                                truncating='post')

    ys_train, ys_test = get_ys(data, train_test_split)

    model = construct_model(max_features, max_len)

    hist = model.fit(
        xs_train_scalar, ys_train,
        batch_size=batch_size,
        validation_data=(xs_test_scalar, ys_test),
        epochs=epochs
    )
    print str(hist)

    # model.save('bidir_1layer_64_emb_pretrained_drop02everywhere.h5')
    model.save('bidir_3layer_16_emb100_drop02everywhere.h5')


def get_xs(data):
    xs_str = data[:, 0]
    xs = np.copy(xs_str)
    for i in range(len(xs)):
        xs[i] = str(xs[i]).decode('utf-8').split(' ')
    # xs = [x.decode('utf-8') for x in xs]
    return xs, xs_str


def test_model_on_labeled_data():
    data = get_data('data/cleaned_data.csv')

    xs, xs_str = get_xs(data)
    best_words_set = data_util.construct_good_set(xs, max_features, 0)
    # pretrained_embedding_vocab = get_ruwiki_embedding_dict()
    # data_util.sentences_to_predefined_scalars(xs, best_words_set, pretrained_embedding_vocab)
    data_util.sentences_to_scalars_loaded_dict(xs, best_words_set)

    xs_scalar = keras.preprocessing.sequence.pad_sequences(xs, maxlen=max_len, padding='post',
                                                                 truncating='post')

    ys = data[:, 1]
    ys = [0 if y == -1 else y for y in ys]
    model = load_model('bidir_2layer_64_emb100_drop02everywhere.h5')
    result = [round(r[0]) for r in model.predict(xs_scalar)]
    visual = zip(xs_str, result)
    predicted_and_true = zip(result, ys)
    pos_cnt = reduce(lambda a, b: a + (1 if b[0] == b[1] and b[0] == 1 else 0), predicted_and_true, 0)
    neg_cnt = reduce(lambda a, b: a + (1 if b[0] == b[1] and b[0] == 0 else 0), predicted_and_true, 0)
    print pos_cnt, neg_cnt
    neg_tot = reduce(lambda a, b: a + (1 if b[0] == 0 else 0), predicted_and_true, 0)
    pos_tot = reduce(lambda a, b: a + (1 if b[0] == 1 else 0), predicted_and_true, 0)
    print pos_tot, neg_tot
    print pos_cnt / float(pos_tot) * 100, neg_cnt / float(neg_tot) * 100

def test_model_on_unlabeled_data():
    data = get_data('data/cleaned_data_ok.csv')

    xs, xs_str = get_xs(data)
    # xs = [x.decode('utf-8') for x in xs]
    best_words_set = data_util.construct_good_set(xs, max_features, 0)
    # pretrained_embedding_vocab = get_ruwiki_embedding_dict()
    # data_util.sentences_to_predefined_scalars(xs, best_words_set, pretrained_embedding_vocab)
    data_util.sentences_to_scalars_loaded_dict(xs, best_words_set)

    xs_scalar = keras.preprocessing.sequence.pad_sequences(xs, maxlen=max_len, padding='post',
                                                                 truncating='post')

    model = load_model('bidir_2layer_64_emb100_drop02everywhere.h5')
    result = model.predict(xs_scalar)#[round(r[0]) for r in model.predict(xs_scalar)]
    visual = zip(xs_str, result)
    visual_sorted = list(sorted(visual, key=lambda x:x[1]))
    print 3


def get_ys(data, train_test_split):
    ys = data[:, 1]
    ys = [0 if y == -1 else y for y in ys]
    ys_train = ys[:train_test_split]
    ys_test = ys[train_test_split:]
    return ys_train, ys_test


def get_data(data_path):
    df = pd.read_csv(data_path, delimiter=",")
    data = np.array(df.as_matrix())[:, 1:]
    return data


def construct_model(max_features, max_len):
    model = Sequential()
    # model.add(get_ruwiki_embedding_keras())
    model.add(Embedding(max_features, 100, input_length=max_len))
    model.add(Bidirectional(LSTM(16, dropout=0, recurrent_dropout=0, return_sequences=True)))
    model.add(Bidirectional(LSTM(16, dropout=0, recurrent_dropout=0, return_sequences=True)))
    model.add(Bidirectional(LSTM(16, dropout=0, recurrent_dropout=0)))
    # model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    # model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model

train_model()
# test_model_on_labeled_data()
# test_model_on_unlabeled_data()