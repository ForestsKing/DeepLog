import numpy as np


def str2list(x):
    x = x.replace('[', '').replace(']', '').replace('\n', ' ').split(' ')
    x = [i for i in x if i != '']
    x = np.array(list(map(np.float, x)))
    return x