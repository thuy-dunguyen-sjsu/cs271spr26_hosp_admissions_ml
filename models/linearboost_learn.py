from linearboost import LinearBoostClassifier as lbc
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from data_encoder import format_data
import shap
import warnings
import logging

logger = logging.getLogger("demo")
warnings.filterwarnings('ignore')


def train_lb(filename='dataset2', optimize=True, random_state=0, params=None):
    logger.info("Begin training LinearBoost model on dataset")
    labels, X_train, X_test, y_train, y_test = format_data(filename, random_state=random_state)
    param_dist = {
        'n_estimators': [10, 15, 50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.3, 0.4, 0.7, 1],
        'algorithm': ['SAMME.R', 'SAMME'],
        'boosting_type': ['adaboost', 'gradient'],
        'early_stopping': [True, False],
        'shrinkage': [0.1, 0.5, 0.8, 0.9, 1],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
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
        random_search = RandomizedSearchCV(lb, param_distributions=param_dist, n_iter=100, scoring='f1', random_state=42, verbose=3)
        random_search.fit(X_train, np.ravel(y_train))

        best_params = random_search.best_params_

        best_model = random_search.best_estimator_
        print(f"Best Hyperparameters: {best_params}")
        logger.info("LinearBoost Hyperparameter optimization complete")
    elif params:
        p = "P"
        logger.info("Modeling with preexisting params")
        best_model = lbc(**params)
        best_model.fit(X_train, y_train)
        best_params = params
    else:
        p = "D"
        logger.info("Modeling with default params")
        best_model = lb
        best_model.fit(X_train, y_train)
        best_params = "Defaults"

    bst = best_model.fit(X_train, np.ravel(y_train))

    preds = best_model.predict(X_test)

    preds = np.round(preds)

    np.savetxt("predictions/" + filename + "_preds_lb.csv", preds, delimiter=",")

    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print('Accuracy of the LinearBoost model is:', accuracy*100)
    mess = "{0}'{1}  >>> RS {2}; Accuracy: {3}; F1: {4}\n".format(p, best_params, random_state, accuracy, f1)
    logger.info(mess)
    with open("params/" + filename + "_params_lb.txt", "a") as text_file:
        text_file.write(mess)



