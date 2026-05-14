import sys
sys.path.insert(0, "./routers")

import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from data_encoder import format_data
import warnings
import os
import logging
from progress import RScv
from tqdm.auto import tqdm
import time

logger = logging.getLogger("demo")
warnings.filterwarnings('ignore')


def train_xgb(filename='dataset2', optimize=True, random_state=0, params=None):
    tqdm.write("Begin training XGBoost model on dataset")
    logger.info("Begin training XGBoost model on dataset")
    labels, X_train, X_test, y_train, y_test = format_data(filename, random_state=random_state)
    #
    param_dist = {
        'eta': [0.01, 0.1, 0.3, 0.7, 1],
        'max_depth': [3, 6, 8, 10, 15, 20],
        # 'colsample_bylevel': [0.01, 0.1, 0.3, 0.6, 0.8, 1.0],
        'colsample_bytree': [0.01, 0.1, 0.3, 0.6, 0.8, 1.0],
        # 'eval_metric': ['auc'],
        # 'objective': ["binary:logistic"]
        'eval_metric': ['rmse', "auc"],
        'objective': ["reg:squarederror", "binary:logistic"]
    }

    xgb_model = xgb.XGBClassifier()

    if optimize:
        p = "O"
        tqdm.write("Beginning LinearBoost randomized hyperparameter optimization")
        logger.info("Beginning xgboost randomized hyperparameter optimization")
        start = time.perf_counter()
        random_search = RScv(xgb_model, param_dist=param_dist)
        random_search.fit(X_train, np.ravel(y_train))

        best_params = random_search.best_params_
        end = time.perf_counter() - start
        tqdm.write("XGBoost Hyperparameter optimization complete: Latency {:.6f}s".format(end))
        logger.info("XGBoost Hyperparameter optimization complete: Latency {:.6f}s".format(end))

        start2 = time.perf_counter()
        tqdm.write("Modeling with RandomSearch params")
        logger.info("Modeling with RandomSearch params")
        best_model = xgb.XGBClassifier(**best_params)
        best_model.fit(X_train, y_train)
        end2 = time.perf_counter() - start2
    elif params:
        start2 = time.perf_counter()
        p = "P"
        tqdm.write("Modeling with preexisting params")
        logger.info("Modeling with preexisting params")
        best_model = xgb.XGBClassifier(**params)
        best_model.fit(X_train, y_train)
        best_params = params
        end2 = time.perf_counter() - start2
    else:
        start2 = time.perf_counter()
        p = "D"
        tqdm.write("Modeling with default params")
        logger.info("Modeling with default params")
        best_model = xgb_model
        best_model.fit(X_train, y_train)
        best_params = "Defaults"
        end2 = time.perf_counter() - start2

    preds = np.round(best_model.predict(X_test))

    # np.savetxt("predictions/" + filename + "_preds_xgb.csv", preds, delimiter=",")

    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    mess = "{}'{}  >>> RS {}; Accuracy: {:.6f}; F1: {:.6f}, latency: {:.6f}s\n".format(p, best_params, random_state,
                                                                                       accuracy, f1, end2)
    tqdm.write(mess)
    logger.info(mess)

    # try:
    #     os.mkdir("./params")
    # except OSError as e:
    #     pass
    #
    # with open("params/" + filename + "_params_lb.txt", "a") as text_file:
    #     text_file.write(mess)


