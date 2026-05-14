import sys
sys.path.insert(0, "./routers")
sys.path.insert(0, "./models")

from sklearn.linear_model import LogisticRegression as lr
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from progress import RScv
from data_encoder import format_data
import warnings
import json
import logging
import os
from tqdm.auto import tqdm
import time

logger = logging.getLogger("demo")
warnings.filterwarnings('ignore')


def opt_lr(filename='dataset2', iterations=None, train_size=None):
    param_dist = {
        'l1_ratio': [1.0],
        'fit_intercept': [True, False],
        'C': np.logspace(-4, 4, 6).tolist(),
        'solver': ['liblinear'],
        'max_iter': [10, 50, 100]
    }
    param_list = {
        'l1_ratio': [],
        'fit_intercept': [],
        'C': [],
        'solver': [],
        'max_iter': [],
        'score': []
    }
    param_final = {
        'l1_ratio': [],
        'fit_intercept': [],
        'C': [],
        'solver': [],
        'max_iter': [],
    }

    lr_model = lr()

    logger.info("Beginning LogisticRegression hyperparameter optimization on subset")
    start = time.perf_counter()
    for x in range(0, iterations):
        labels, X_train, X_test, y_train, y_test = format_data(filename, test_size=train_size,
                                                               train_size=train_size, stratify=True)
        random_search = RScv(lr_model, param_dist=param_dist)
        random_search.fit(X_train, y_train)
        best_params = random_search.best_params_
        for (k, v) in best_params.items():
            param_list[k].append(v)
        logger.info("Iter:{}'{}  >>> Score: {:.6f}\n".format(x, best_params, random_search.best_score_))
        tqdm.write("Iter:{}'{}  >>> Score: {:.6f}\n".format(x, best_params, random_search.best_score_))

    for (k, v) in param_list.items():
        if k in ("accuracy", "f1", "score"):
            pass
        elif type(v[0]) in (str, bool):
            param_final[k] = max(set(v), key=v.count)
        elif type(v[0]) in (float, int):
            param_final[k] = type(v[0])(sum(v) / len(v))
    end = time.perf_counter() - start

    logger.info("Final Parameters with latency {:.6f}s: {}".format(end, json.dumps(param_final, indent=4)))
    mess = "{0}\n\n{1}\n\n".format(json.dumps(param_list, indent=4), json.dumps(param_final, indent=4))
    tqdm.write("Final Parameters with latency {:.6f}s: {}".format(end, json.dumps(param_final, indent=4)))

    # try:
    #     os.mkdir("./params")
    # except OSError as e:
    #     pass
    #
    # with open("params/" + filename + "_opt_lr.txt", "w") as text_file:
    #     text_file.write(mess)

    return param_final




