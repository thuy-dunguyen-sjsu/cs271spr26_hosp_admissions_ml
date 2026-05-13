import shap


def shapper(bst, X_train):
    print("Beginning shap analysis")
    # Uses Shap to analyze the models algorithm for weighing features
    explainer = shap.Explainer(bst.predict, X_train)
    shap_values = explainer(X_train, max_evals=2000)
    shap.plots.beeswarm(shap_values)

