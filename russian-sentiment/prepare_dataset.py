# coding=utf-8
import string

import re
import sklearn
import pandas as pd
import pymorphy2

morph = pymorphy2.MorphAnalyzer()
regex = re.compile('[%s\w]' % re.escape(string.punctuation))
i = 0


def text_cleaner(text, is_utf=True):
    global i
    i += 1
    if i % 1000 == 0:
        print i

    text = text.lower()
    if not is_utf:
        text = text.decode('utf-8')
    text = regex.sub('', text)
    res = [morph.parse(word)[0].normal_form for word in text.split()]

    return ' '.join(res)


def prepare_twitter_dataset():
    positives = pd.read_csv('data/positive.csv', sep=';', header=None)
    negatives = pd.read_csv('data/negative.csv', sep=';', header=None)

    dataset = pd.concat([positives, negatives])
    dataset = dataset[[3, 4]]
    dataset.columns = ['text', 'label']

    dataset['text'] = dataset['text'].apply(text_cleaner, is_utf=False)

    dataset.to_csv('data/cleaned_data.csv', encoding='utf-8')


def prepare_ok_dataset():
    dataset = pd.read_csv('data/train_content.csv', sep='\t', header=None, encoding='utf-8')
    dataset.columns = ['text']

    dataset['text'] = dataset['text'].apply(text_cleaner)

    dataset.to_csv('data/cleaned_data_ok.csv', encoding='utf-8')


prepare_ok_dataset()
