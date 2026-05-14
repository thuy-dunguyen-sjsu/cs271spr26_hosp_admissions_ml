import sys
sys.path.insert(0, "./routers")

from main import route_model
from hp_optimize import route
import logging
import argparse
from tqdm.auto import tqdm as tq


logger = logging.getLogger("demo")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("demo.log")
logger.addHandler(file_handler)
# logger.addHandler(logging.StreamHandler(sys.stdout))


def parse():
    parse = argparse.ArgumentParser()
    parse.add_argument("-c", "--complete", action='store_true')
    args = parse.parse_args()
    if args.complete:
        return "complete"
    else:
        return "subset"


def subset():
    logger.info("******************************DEMO SUBSET OPTIMIZER INITIATED**************************************")
    filename = "dataset3"
    iterations = 10
    train_size = 0.01
    logger.info("Filename:{}, iterations:{}, train_size:{}".format(filename, iterations, train_size))

    # data_enc(filename)

    for m in ["lr", "lb", "xgb"]:
        tq.write(m)
        params = route(m, filename, iterations, train_size)
        route_model(m, filename, 42, False, params)

    logger.info("*******************************DEMO COMPLETE*************************************************")


def rand_opt():
    logger.info("******************************DEMO RANDOM OPTIMIZER INITIATED**************************************")
    filename = "dataset3"
    random_state = 42
    logger.info("Filename:{}, random_state:{}".format(filename, random_state))
    for m in ["lr", "lb", "xgb"]:
        tq.write(m)
        route_model(m, filename, 42, True, None)

    logger.info("*******************************DEMO COMPLETE*************************************************")


if __name__ == "__main__":
    demo = parse()
    match demo:
        case "subset":
            subset()
        case "complete":
            rand_opt()
