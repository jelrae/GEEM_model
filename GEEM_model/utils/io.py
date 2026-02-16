from numpy.random import Generator, PCG64
from scipy.stats import binom, gamma
import pickle
import os


def set_seeds(seed):
    numpy_randomGen = Generator(PCG64(seed))
    scipy_randomGen = binom
    scipy_randomGen.random_state = numpy_randomGen
    return numpy_randomGen, scipy_randomGen


def create_dirs(fps):
    """
    :param fps: a list of strings which contain the filepaths desired to create
    :return:
    """

    for p in fps:
        os.makedirs(p, exist_ok=True)
        print("The directory " + p + " is created!")


def load_param(fp):
    with open(fp, 'rb') as file:
        ep = pickle.load(file)
    return ep


def save_params(fp, fn, p):
    print("Current working directory:", os.getcwd())
    create_dirs([fp])
    with open(fp + "/" + fn, "wb") as pickie:
        pickle.dump(p, pickie)
