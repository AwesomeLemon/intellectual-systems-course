import cPickle


def store_obj(items, name):
    cPickle.dump(items, open(name, 'wb'))


def retrieve_obj(name):
    return cPickle.load(open(name, 'rb'))
