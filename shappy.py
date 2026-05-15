import sys
sys.path.insert(0, "./routers")
sys.path.insert(0, "./models")

import shap
import logging
import xgboost as xgb
from linearboost import LinearBoostClassifier as lbc
from sklearn.linear_model import LogisticRegression as lr
import os
from sklearn.metrics import accuracy_score, f1_score
from model_params import op_p
from data_encoder import format_data
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger("noise")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("params/noise.log")
logger.addHandler(file_handler)
logger.addHandler(logging.StreamHandler(sys.stdout))

filename = "dataset2"

def analyze(bst, X_train, model):
    print("Beginning shap analysis")
    fig = plt.figure()
    save_path = 'figures/shap_beeswarm_{}_plot.png'.format(model)
    try:
        os.mkdir("./figures")
    except OSError as e:
        pass
    # Uses Shap to analyze the models algorithm for weighing features
    explainer = shap.Explainer(bst.predict, X_train)
    shap_values = explainer(X_train, max_evals=2000)
    shap.plots.beeswarm(shap_values, show=False)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

lr_params = op_p["lr"]
lb_params = op_p["lb"]
xgb_params = op_p["xgb"]

lr_model = lr(**lr_params)
xgb_model = xgb.XGBClassifier(**xgb_params)
lb_model = lbc(**lb_params)

labels, X_train, X_test, y_train, y_test = format_data(filename, random_state=42)
index = list(range(X_train.shape[0]))

X_pd = pd.DataFrame(X_train, columns=labels, index=index)

models = {'lr_model': lr_model, 'xgb_model': xgb_model, 'lb_model': lb_model}

for (mn, mm) in models.items():
    m_fit = mm.fit(X_pd, y_train)
    analyze(mm, X_pd, mn)

