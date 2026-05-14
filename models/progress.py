from tqdm import tqdm
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RandomizedSearchCV

pbar = tqdm(total=100, miniters=1)

def tqdm_scorer(y_true, y_pred):
    pbar.update(1) # Update bar for every fit
    return f1_score(y_true, y_pred)

def RScv(model, param_dist):
    global pbar
    pbar = tqdm(total=2*counter(param_dist), miniters=1)
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=25, scoring=make_scorer(tqdm_scorer),
                                       random_state=42, cv=2)
    return random_search

def counter(d):
    x = 1
    for (k, v) in d.items():
        x = x*len(v)
    if x > 50:
        x = 50
    return x