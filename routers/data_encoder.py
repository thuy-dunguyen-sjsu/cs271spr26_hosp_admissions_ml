import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import logging
import os

logger = logging.getLogger(__name__)

def data_enc(filename):
    # Reads in tsv file and formats data
    logger.info("Processing dataset with encodings from datasets/" + filename + ".csv")
    data = np.loadtxt("datasets/" + filename + ".csv", delimiter=',', dtype=np.object_)
    # data = np.genfromtxt(filename+'.csv', delimiter=',', dtype=None)

    labels = np.concatenate((data[0, 0:11], data[0, 12:], ["disposition"]))
    X = np.concatenate((data[1:, 0:11], data[1:, 12:]), axis=1)
    y = [[1] if x == 'Admit' else [0] for x in data[1:, 11]]

    encoded_X = np.empty(X.shape)
    #
    for c in range(len(labels)-1):
        encoder = LabelEncoder()
        encoded_X[:, c] = encoder.fit_transform(X[:, c])

    encoded = np.concatenate((encoded_X, y), axis=1)
    encoded = np.concatenate(([labels], encoded), axis=0)

    try:
        os.mkdir("../datasets_encoded")
    except OSError as e:
        pass

    np.savetxt("datasets_encoded/" + filename + "_encode.csv", encoded, delimiter=",", fmt='%s')
    logger.info(filename + " encoded and saved at" + "datasets_encoded/" + filename + "_encode.csv")


def format_data(filename, test_size=None, train_size=None, random_state=None, stratify=None):
    data = np.loadtxt("datasets_encoded/" + filename + "_encode.csv", delimiter=',', skiprows=1)
    labels_data = np.loadtxt("datasets_encoded/" + filename + "_encode.csv", delimiter=',', max_rows=1, dtype=np.object_)
    nrows, ncols = data.shape
    labels = labels_data[0:ncols-2]
    X = data[:, 0:ncols-2]
    y = [x for x in data[:, ncols-1]]

    if stratify:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size,
                                                        random_state=random_state, stratify=stratify)
    logger.info(filename + " split into training and test sets with train_size: {} and random_state: {}".format(
        train_size, random_state))

    return labels, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    filename = "dataset3"
    data_enc(filename)
    labels, X_train, X_test, y_train, y_test = format_data(filename, 0.25, 0)
