import sys
sys.path.insert(0, "./tests")
from demo import subset, rand_opt
from latency import run as run_l
from noise import run as run_n


run_l()

run_n()

subset()

rand_opt()

