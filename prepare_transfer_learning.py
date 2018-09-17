"""
    Function to be called when we wanna combine the training(and test) data for both LCQuAD and QALD in different manner.

    - transfer_a : train on lcquad+qald; test on qald
    - transfer_b : train on lcquad; test on qaldtest
    - transfer_c : train on lcquad; test on qald test+train     (dontneed)
    - transfer_d : train on lcquad+qald; test on lcquad test

"""

import os
import sys
import json
import numpy as np

import network as n


DATA_DIR = './data/data/%(dataset)s/'
DEFAULT_METHOD = 'transfer-b'
np.random.seed(42)

def transfer_a(DATASET_SPECIFIC_DATA_DIR=DATA_DIR):
    """
        This function tries to see if the files needed are already in there or not.
        If they are,
            it will return filename and index which can be used by createdataset and models
        If not,
            it will open LCQuAD, and merge it with qald train, and test.
            save file, save index
    :return: filename: str; index: int
    """

    try:
        raise IOError
    except IOError:

        # Open files.

        lcquad_json = json.load(open(os.path.join(DATASET_SPECIFIC_DATA_DIR % {'dataset': 'lcquad'}, "id_big_data.json")))
        qald_train_json = json.load(open(os.path.join(DATASET_SPECIFIC_DATA_DIR % {'dataset':'qald'}, "qald_id_big_data_train.json")))
        qald_train_json = qald_train_json[:int(7.0*len(qald_train_json)/8.0)]
        qald_valid_json = qald_train_json[int(7.0*len(qald_train_json)/8.0):]
        # qald_test_json = json.load(open(os.path.join(n.DATASET_SPECIFIC_DATA_DIR % {'dataset':'qald'}, "qald_id_big_data_test.json")))

        # Combine files
        a = lcquad_json + qald_train_json
        np.random.shuffle(a)

        combined_json = a + qald_valid_json

        # store the combined file
        json.dump(combined_json,

                  open(os.path.join(DATASET_SPECIFIC_DATA_DIR % {'dataset': 'transfer-a'}, 'combined.json'), 'w+'))

        index = len(lcquad_json) + len(qald_train_json) - 1

        # store index

        f = open(os.path.join(DATASET_SPECIFIC_DATA_DIR % {'dataset': 'transfer-a'}, 'index'), 'w+')
        f.write(str(index))
        f.close()

        return 'combined.json', index


def transfer_b(DATASET_SPECIFIC_DATA_DIR=DATA_DIR):
    """
        This function tries to see if the files needed are already in there or not.
        If they are,
            it will return filename and index which can be used by createdataset and models
        If not,
            it will open LCQuAD, and merge it with qald test.
            save file, save index
    :return: filename: str; index: int
    """
    try:
        raise IOError
    except IOError:

        # Open files.

        lcquad_json = json.load(open(os.path.join(DATASET_SPECIFIC_DATA_DIR % {'dataset': 'lcquad'}, "id_big_data.json")))
        lcquad_train_json = lcquad_json[:int(0.7*len(lcquad_json))]
        lcquad_valid_json = lcquad_json[int(0.7*len(lcquad_json)):int(0.8*len(lcquad_json))]
        qald_train_json = json.load(open(os.path.join(DATASET_SPECIFIC_DATA_DIR % {'dataset':'qald'}, "qald_id_big_data_train.json")))

        # Combine files
        a = lcquad_train_json + qald_train_json
        np.random.shuffle(a)
        combined_json = a + lcquad_valid_json

        # store the combined file
        json.dump(combined_json,

                  open(os.path.join(DATASET_SPECIFIC_DATA_DIR % {'dataset': 'transfer-b'}, 'combined.json'), 'w+'))

        index = len(lcquad_train_json) + len(qald_train_json) - 1

        # store index

        f = open(os.path.join(DATASET_SPECIFIC_DATA_DIR % {'dataset': 'transfer-b'}, 'index'), 'w+')
        f.write(str(index))
        f.close()

        return 'combined.json', index



def transfer_c(DATASET_SPECIFIC_DATA_DIR=DATA_DIR):
    """
        This function tries to see if the files needed are already in there or not.
        If they are,
            it will return filename and index which can be used by createdataset and models
        If not,
            it will open LCQuAD, and merge it with qald train, and test.
            save file, save index
    :return: filename: str; index: int
    """

    try:
        raise IOError
    except IOError:

        # Open files.

        lcquad_json = json.load(open(os.path.join(DATASET_SPECIFIC_DATA_DIR % {'dataset': 'lcquad'}, "id_big_data.json")))
        qald_train_json = json.load(open(os.path.join(DATASET_SPECIFIC_DATA_DIR % {'dataset':'qald'}, "qald_id_big_data_train.json")))
        qald_test_json = json.load(open(os.path.join(DATASET_SPECIFIC_DATA_DIR % {'dataset':'qald'}, "qald_id_big_data_test.json")))

        # Combine files
        combined_json = lcquad_json + qald_train_json + qald_test_json

        # store the combined file
        json.dump(combined_json,

                  open(os.path.join(DATASET_SPECIFIC_DATA_DIR % {'dataset': 'transfer-c'}, 'combined.json'), 'w+'))

        index = len(lcquad_json) - 1

        # store index

        f = open(os.path.join(DATASET_SPECIFIC_DATA_DIR % {'dataset': 'transfer-c'}, 'index'), 'w+')
        f.write(str(index))
        f.close()

        return 'combined.json', index


if __name__ == '__main__':

    # Get method
    method = sys.argv[1].strip().lower()

    while True:

        try:
            assert method in ['transfer-a', 'transfer-b', 'transfer-r']
            break
        except AssertionError:
            method = raw_input("Did not understand method. Please type it again: ")

    if method == 'transfer-a':
        print(transfer_a())
    elif method == 'transfer-b':
        print(transfer_b())
    elif method == 'transfer-c':
        print(transfer_c())


