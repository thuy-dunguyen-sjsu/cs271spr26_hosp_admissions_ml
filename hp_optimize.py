import sys
sys.path.insert(0, "./hyperp_opt")

from xgb_hp_opt import opt_xgb
from lr_hp_opt import opt_lr
from lb_hp_opt import opt_lb
import argparse

def parse():
    parse = argparse.ArgumentParser()
    parse.add_argument('model', type=str, default="lr")
    parse.add_argument('filename', type=str, default="dataset2")
    parse.add_argument('iterations', type=int, default=1)
    parse.add_argument('train_size', type=float, default=None)

    args = parse.parse_args()
    model = args.model
    filename = args.filename
    iterations = args.iterations
    train_size = args.train_size
    return model, filename, iterations, train_size

def route(model, filename, iterations, train_size):
    opt_params = None
    match model:
        case "xgb":
            opt_params = opt_xgb(filename=filename, iterations=iterations, train_size=train_size)
        case "lr":
            opt_params = opt_lr(filename=filename,  iterations=iterations, train_size=train_size)
        case "lb":
            opt_params = opt_lb(filename=filename,  iterations=iterations, train_size=train_size)
        case _:
            print("Training could not be done with the expected options.")
    return opt_params


if __name__ == "__main__":
    model, filename, iterations, train_size = parse()
    route(model, filename, iterations, train_size)
