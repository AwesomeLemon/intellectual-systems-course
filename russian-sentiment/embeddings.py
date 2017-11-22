# coding=utf-8
from gensim.models.keyedvectors import KeyedVectors

model = KeyedVectors.load_word2vec_format('ruwikiruscorpora_rusvectores2_my2.bin', binary=True)


def get_ruwiki_embedding_keras():
    return model.get_keras_embedding()


def get_ruwiki_embedding_dict():
    return model.vocab


def adapt_ruwiki_embedding():
    model = KeyedVectors.load_word2vec_format('/home/alex/Downloads/ruwikiruscorpora_rusvectores2.bin', binary=True)
    model.save_word2vec_format('ruwikiruscorpora_rusvectores2_my2.bin', binary=True)
