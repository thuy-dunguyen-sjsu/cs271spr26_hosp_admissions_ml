from linearboost import LinearBoostClassifier as lbc
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from data_encoder import format_data
import shap
import warnings
import json
import logging
import os


logger = logging.getLogger("demo")

warnings.filterwarnings('ignore')


def opt_lb(filename='dataset2', iterations=None, train_size=None):
    param_dist = {
        'n_estimators': [10, 15, 50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.3, 0.4, 0.7, 1],
        'algorithm': ['SAMME.R'],
        'boosting_type': ['adaboost'],
        'early_stopping': [True, False],
        'shrinkage': [0.1, 0.5, 0.8, 0.9, 1],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }

    param_list = {
        'n_estimators': [],
        'learning_rate': [],
        'algorithm': [],
        'boosting_type': [],
        'early_stopping': [],
        'shrinkage': [],
        'kernel': [],
        'accuracy': [],
        'f1': []
    }

    param_final = {
        'n_estimators': [],
        'learning_rate': [],
        'algorithm': [],
        'boosting_type': [],
        'early_stopping': [],
        'shrinkage': [],
        'kernel': []
    }

    lb = lbc()

    logger.info("Beginning linearboost hyperparameter optimization on subset")
    for x in range(0, iterations):
        labels, X_train, X_test, y_train, y_test = format_data(filename, test_size=train_size,
                                                               train_size=train_size, stratify=True)
        random_search = RandomizedSearchCV(lb, param_distributions=param_dist, n_iter=100, scoring='f1',
                                           random_state=42)
        random_search.fit(X_train, np.ravel(y_train))
        best_params = random_search.best_params_
        best_model = random_search.best_estimator_
        for (k, v) in best_params.items():
            param_list[k].append(v)
        preds = best_model.predict(X_test)
        preds = np.round(preds)
        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        param_list["accuracy"].append(accuracy)
        param_list["f1"].append(f1)
        logger.info("Iter:{0}'{1}  >>> Accuracy: {2}; F1: {3}\n".format(x, best_params, accuracy, f1))

    for (k, v) in param_list.items():
        if k in ("accuracy", "f1"):
            pass
        elif type(v[0]) in (str, bool):
            param_final[k] = max(set(v), key=v.count)
        elif type(v[0]) in (float, int):
            param_final[k] = type(v[0])(sum(v) / len(v))

    logger.info("Final Parameters: {}".format(json.dumps(param_final, indent=4)))
    mess = "{0}\n\n{1}\n\n".format(json.dumps(param_list, indent=4), json.dumps(param_final, indent=4))

    try:
        os.mkdir("./params")
    except OSError as e:
        pass

    with open("params/" + filename + "_opt_lb.txt", "w") as text_file:
        text_file.write(mess)

    return param_final

