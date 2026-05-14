from tqdm import tqdm
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RandomizedSearchCV
from tqdm.auto import tqdm as tq

pbar = tqdm(total=100, miniters=1)
n = 25
c = 2

def tqdm_scorer(y_true, y_pred):
    pbar.update(1) # Update bar for every fit
    return f1_score(y_true, y_pred)

def RScv(model, param_dist):
    global pbar
    pbar = tqdm(total=c*counter(param_dist), miniters=1)
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n,
                                       scoring=make_scorer(tqdm_scorer), random_state=42, cv=c)
    return random_search

def counter(d):
    x = 1
    for (k, v) in d.items():
        x = x*len(v)
    if x*c > n*c:
        x = n
    return x