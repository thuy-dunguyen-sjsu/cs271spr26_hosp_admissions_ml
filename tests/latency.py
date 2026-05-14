import sys
sys.path.insert(0, "./routers")
import logging
import time
import xgboost as xgb
from linearboost import LinearBoostClassifier as lbc
from sklearn.linear_model import LogisticRegression as lr
from data_encoder import format_data
import json

logger = logging.getLogger("latency")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("params/latency.log")
logger.addHandler(file_handler)
logger.addHandler(logging.StreamHandler(sys.stdout))

def run():
    lr_model = lr(solver="liblinear")
    xgb_model = xgb.XGBClassifier()
    lb_model = lbc()
    labels, X_train, X_test, y_train, y_test = format_data(filename="dataset2")
    latency = {}
    models = ['lr_model', 'xgb_model', 'lb_model']
    ms = [lr_model, xgb_model, lb_model]
    iterations = [1, 10, 50, 100, 500]

    logger.info("**********************************Begin Latency Test*****************************")

    for i in iterations:
        for m in range(len(models)):
            start = time.perf_counter()
            for x in range(i):
                ms[m].fit(X_train, y_train)
            end = time.perf_counter() - start
            latency[models[m]] = end
        logger.info("Iterations: {}".format(i))
        logger.info(json.dumps(latency, indent=4))

    logger.info("**********************************Complete Latency Test*****************************")

if __name__ == "__main__":
    run()