import pickle
import pandas as pd
import os


def pickle_to_file(file_name, data, protocol=pickle.HIGHEST_PROTOCOL):
    assert os.path.isfile(file_name)
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol)


def unpickle_from_file(file_name):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)