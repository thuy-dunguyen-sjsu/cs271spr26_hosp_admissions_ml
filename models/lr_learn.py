import sys
sys.path.insert(0, "./routers")

from sklearn.linear_model import LogisticRegression as lr
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from data_encoder import format_data
import warnings
import logging
import os
from progress import RScv
from tqdm.auto import tqdm
import time

logger = logging.getLogger("demo")
warnings.filterwarnings('ignore')


def train_lr(filename='dataset2', optimize=True, random_state=0, params=None):
    tqdm.write("Begin training LogisticRegression model on dataset")
    logger.info("Begin training LogisticRegression model on dataset")
    labels, X_train, X_test, y_train, y_test = format_data(filename, random_state=random_state)

    # param_dist = {'copy_X': [True, False],
    #               'fit_intercept': [True, False],
    #               'n_jobs': [1, 5, 10, 15, None],
    #               'positive': [True, False]
    #               }

    param_dist = {
        'l1_ratio': [1.0],
        'fit_intercept': [True, False],
        'C': np.logspace(-4,4,6).tolist(),
        'solver': ['liblinear'],
        'max_iter': [10, 50, 100]
    }

    lr_model = lr()

    if optimize:
        p = "O"
        logger.info("Beginning LogisticRegression randomized hyperparameter optimization")

        start = time.perf_counter()
        random_search = RScv(lr_model, param_dist=param_dist)
        end = time.perf_counter() - start

        start2 = time.perf_counter()
        random_search.fit(X_train, y_train)

        best_params = random_search.best_params_
        best_model = random_search.best_estimator_
        end2 = time.perf_counter() - start2

        logger.info("LogisticRegression Hyperparameter optimization complete: Latency {:.6f}s".format(end))

    elif params:
        start2 = time.perf_counter()
        p = "P"
        tqdm.write("Modeling with preexisting params")
        logger.info("Modeling with preexisting params")
        best_model = lr(**params)
        best_model.fit(X_train, y_train)
        best_params = params
        end2 = time.perf_counter() - start2
    else:
        start2 = time.perf_counter()
        p = "D"
        tqdm.write("Modeling with default params")
        logger.info("Modeling with default params")
        best_model = lr_model
        best_model.fit(X_train, y_train)
        best_params = "Defaults"
        end2 = time.perf_counter() - start2

    # bst = best_model.fit(X_train, y_train)

    preds = np.round(best_model.predict(X_test))

    np.savetxt("predictions/" + filename + "_preds_lr.csv", preds, delimiter=",")

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



    # #Uses Shap to analyze the models algorithm for weighing features
    # explainer = shap.Explainer(bst.predict, X_train)
    # shap_values = explainer(X_train, max_evals=2000)
    # shap.plots.beeswarm(shap_values)

