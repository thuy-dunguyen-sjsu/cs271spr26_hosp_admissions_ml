import sys
sys.path.insert(0, "./models")

from xgb_learn import train_xgb
from lr_learn import train_lr
from linearboost_learn import train_lb
import argparse


def parse():
    parse = argparse.ArgumentParser()
    parse.add_argument('model', type=str, default="lr")
    parse.add_argument('filename', type=str, default="dataset2")
    parse.add_argument('random_state', type=int, default=0)
    parse.add_argument('-o', '--optimize', action='store_true')


    args = parse.parse_args()
    model = args.model
    filename = args.filename
    random_state = args.random_state
    opt = args.optimize

    return model, filename, random_state, opt

def route_model(model, filename, random_state, opt, params=None):
    match model:
        case "xgb":
            train_xgb(filename=filename, optimize=opt, random_state=random_state, params=params)
        case "lr":
            train_lr(filename=filename, optimize=opt, random_state=random_state, params=params)
        case "lb":
            train_lb(filename=filename, optimize=opt, random_state=random_state, params=params)
        case _:
            print("Training could not be done with the expected options.")


if __name__ == "__main__":
    model, filename, random_state, opt = parse()
    route_model(model, filename, random_state, opt)



