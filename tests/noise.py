import numpy as np
import sys
sys.path.insert(0, "./routers")
import logging
import xgboost as xgb
from linearboost import LinearBoostClassifier as lbc
from sklearn.linear_model import LogisticRegression as lr
from data_encoder import format_data
import json
from sklearn.metrics import accuracy_score, f1_score


logger = logging.getLogger("noise")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("params/noise.log")
logger.addHandler(file_handler)
logger.addHandler(logging.StreamHandler(sys.stdout))

noise = [5, 10, 20]


def test(noise):
    logger.info("**********************************Begin Noise Test*****************************")
    # lr_params = {'solver': 'liblinear', 'max_iter': 10, 'l1_ratio': 1.0, 'fit_intercept': False, 'C': 6.309573444801943}
    lr_params = {"l1_ratio": 1.0, "fit_intercept": True, "C": 3077.880422323208, "solver": "liblinear", "max_iter": 45}
    lb_params = {'shrinkage': 0.9, 'n_estimators': 50, 'learning_rate': 0.1, 'early_stopping': True}
    xgb_params = {'objective': 'binary:logistic', 'max_depth': 6, 'eval_metric': 'rmse', 'eta': 0.1, 'colsample_bytree': 1.0}
    lr_model = lr(**lr_params)
    xgb_model = xgb.XGBClassifier(**xgb_params)
    lb_model = lbc(**lb_params)
    lr_model_ns = lr(**lr_params)
    xgb_model_ns = xgb.XGBClassifier(**xgb_params)
    lb_model_ns = lbc(**lb_params)
    labels, X_train, X_test, y_train, y_test = format_data(filename="dataset2")
    models = ['lr_model', 'xgb_model', 'lb_model']
    ms = [lr_model, xgb_model, lb_model]
    ms_ns = [lr_model_ns, xgb_model_ns, lb_model_ns]

    # data_shape = X_train.shape
    # X_train_noise = X_train
    # noise_size = int(noise*data_shape[0]*data_shape[1])
    # random_indices = np.random.choice(data_shape, noise_size)
    # noise_sample = np.random.choice([X_train.min(), X_train.max()], noise_size)
    # X_train_noise.flat[random_indices] = noise_sample

    X_train_noise = np.random.normal(X_train, noise)

    for x in range(len(models)):
        og = ms[x]
        ns = ms_ns[x]
        og.fit(X_train, y_train)
        ns.fit(X_train_noise, y_train)

        preds_og = np.round(og.predict(X_test))
        preds_ns = np.round(ns.predict(X_test))

        accuracy_og = accuracy_score(y_test, preds_og)
        f1_og = f1_score(y_test, preds_og)

        accuracy_ns = accuracy_score(y_test, preds_ns)
        f1_ns = f1_score(y_test, preds_ns)

        logger.info(models[x] + " Noise Deviation: {}".format(noise))
        logger.info("Accuracy OG: {:.6f}; F1 OG: {:.6f}".format(accuracy_og, f1_og))
        logger.info("Accuracy NS: {:.6f}; F1 NS: {:.6f}".format(accuracy_ns, f1_ns))

    logger.info("**********************************Complete Noise Test*****************************")

for n in noise:
    test(n)