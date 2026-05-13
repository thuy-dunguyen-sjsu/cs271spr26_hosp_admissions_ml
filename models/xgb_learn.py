import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from data_encoder import format_data
import shap
import warnings

import logging

logger = logging.getLogger("demo")
warnings.filterwarnings('ignore')


def train_xgb(filename='dataset2', optimize=True, random_state=0, params=None):
    logger.info("Begin training XGBoost model on dataset")
    labels, X_train, X_test, y_train, y_test = format_data(filename, random_state=random_state)
    #
    param_dist = {
        'eta': [0.01, 0.1, 0.3, 0.7, 1],
        'max_depth': [3, 6, 8, 10, 15, 20],
        'colsample_bylevel': [0.01, 0.1, 0.3, 0.6, 0.8, 1.0],
        'colsample_bytree': [0.01, 0.1, 0.3, 0.6, 0.8, 1.0],
        # 'eval_metric': ['auc'],
        # 'objective': ["binary:logistic"]
        'eval_metric': ['rmse'],
        'objective': ["reg:squarederror"]
    }

    xgb_model = xgb.XGBClassifier()

    if optimize:
        p = "O"
        logger.info("Beginning xgboost randomized hyperparameter optimization")
        random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=100, scoring='f1',
                                           random_state=42, verbose=3)
        random_search.fit(X_train, y_train)

        best_params = random_search.best_params_

        best_model = random_search.best_estimator_

        logger.info("xgboost Hyperparameter optimization complete")

        print(f"Best Hyperparameters: {best_params}")
    elif params:
        p = "P"
        logger.info("Modeling with preexisting params")
        best_model = xgb.XGBClassifier(**params)
        best_model.fit(X_train, y_train)
        best_params = params
    else:
        p = "D"
        logger.info("Modeling with default params")
        best_model = xgb_model
        best_model.fit(X_train, y_train)
        best_params = "Defaults"

    bst = best_model.fit(X_train, y_train)

    preds = best_model.predict(X_test)

    preds = np.round(preds)

    np.savetxt("predictions/" + filename + "_preds_xgb.csv", preds, delimiter=",")

    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print('Accuracy of the xgb model is:', accuracy*100)
    mess = "{0}'{1}  >>> RS {2}; Accuracy: {3}; F1: {4}\n".format(p, best_params, random_state, accuracy, f1)
    logger.info(mess)
    with open("params/" + filename + "_params_lb.txt", "a") as text_file:
        text_file.write(mess)


    # # Uses Shap to analyze the models algorithm for weighing features
    # explainer = shap.Explainer(bst.predict, X_train)
    # shap_values = explainer(X_train, max_evals=2000)
    # shap.plots.beeswarm(shap_values)

    # accuracy = best_model.score(X_test, y_test)
    # print(f"Accuracy on test set: {accuracy:.2f}")

