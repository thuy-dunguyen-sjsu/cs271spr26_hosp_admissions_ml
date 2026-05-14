import sys
sys.path.insert(0, "./routers")

from linearboost import LinearBoostClassifier as lbc
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from data_encoder import format_data
import warnings
import logging
import os
from progress import  RScv
from tqdm.auto import tqdm
import time

logger = logging.getLogger("demo")
warnings.filterwarnings('ignore')


def train_lb(filename='dataset2', optimize=True, random_state=0, params=None):
    tqdm.write("Begin training LinearBoost model on dataset")
    logger.info("Begin training LinearBoost model on dataset")
    labels, X_train, X_test, y_train, y_test = format_data(filename, random_state=random_state)
    param_dist = {
        'n_estimators': [10, 15, 50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.4, 0.7, 1],
        'early_stopping': [True, False],
        'shrinkage': [0.8, 0.9, 1],
        # 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }

    # param_dist = {
    #     'n_estimators': [50],
    #     'learning_rate': [0.01],
    #     'algorithm': ['SAMME.R'],
    #     'boosting_type': ['adaboost'],
    #     'early_stopping': [False]
    # }

    lb = lbc()

    # lb.fit(X_train, np.ravel(y_train))
    # bst = lb.predict(X_test)

    if optimize:
        p = "O"
        logger.info("Beginning LinearBoost randomized hyperparameter optimization")
        start = time.perf_counter()
        random_search = RScv(lb, param_dist=param_dist)
        end = time.perf_counter() - start

        start2 = time.perf_counter()
        random_search.fit(X_train, np.ravel(y_train))

        best_params = random_search.best_params_
        best_model = random_search.best_estimator_
        end2 = time.perf_counter() - start2
        logger.info("LinearBoost Hyperparameter optimization complete: Latency {:.6f}s".format(end))
    elif params:
        start2 = time.perf_counter()
        p = "P"
        tqdm.write("Modeling with preexisting params")
        logger.info("Modeling with preexisting params")
        best_model = lbc(**params)
        best_model.fit(X_train, y_train)
        best_params = params
        end2 = time.perf_counter() - start2
    else:
        start2 = time.perf_counter()
        p = "D"
        tqdm.write("Modeling with default params")
        logger.info("Modeling with default params")
        best_model = lb
        best_model.fit(X_train, y_train)
        best_params = "Defaults"
        end2 = time.perf_counter() - start2

    # bst = best_model.fit(X_train, np.ravel(y_train))

    preds = np.round(best_model.predict(X_test))

    np.savetxt("predictions/" + filename + "_preds_lb.csv", preds, delimiter=",")

    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    mess = "{}'{}  >>> RS {}; Accuracy: {:.6f}; F1: {:.6f}, latency: {:.6f}s\n".format(p, best_params, random_state,
                                                                                    accuracy, f1, end2)
    tqdm.write(mess)
    logger.info(mess)

    try:
        os.mkdir("./params")
    except OSError as e:
        pass

    with open("params/" + filename + "_params_lb.txt", "a") as text_file:
        text_file.write(mess)



