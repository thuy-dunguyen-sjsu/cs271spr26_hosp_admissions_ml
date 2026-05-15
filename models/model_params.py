import numpy as np

mp = {
    "lr": {
        'l1_ratio': [1.0],
        'fit_intercept': [True, False],
        'C': np.logspace(-4,4,6).tolist(),
        'solver': ['liblinear'],
        'max_iter': [10, 50, 100]
        },
    "lb": {
        'n_estimators': [10, 15, 50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.4, 0.7, 1],
        'early_stopping': [True, False],
        'shrinkage': [0.8, 0.9, 1.0],
        # 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
        },
    "xgb": {
        'eta': [0.01, 0.1, 0.3, 0.7, 1],
        'max_depth': [3, 6, 8, 10, 15, 20],
        # 'colsample_bylevel': [0.01, 0.1, 0.3, 0.6, 0.8, 1.0],
        'colsample_bytree': [0.01, 0.1, 0.3, 0.6, 0.8, 1.0],
        # 'eval_metric': ['auc'],
        # 'objective': ["binary:logistic"]
        'eval_metric': ['rmse', "auc"],
        'objective': ["reg:squarederror", "binary:logistic"]
        }
}


op_p = {
    "lr": {"l1_ratio": 1.0, "fit_intercept": True, "C": 5.079356619690776, "solver": "liblinear", "max_iter": 45},
    "lb": {'shrinkage': 0.9, 'n_estimators': 50, 'learning_rate': 0.1, 'early_stopping': True},
    "xgb": {'objective': 'binary:logistic', 'max_depth': 6, 'eval_metric': 'rmse', 'eta': 0.1, 'colsample_bytree': 1.0}
}