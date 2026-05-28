from fast_reid.fastreid.data.datasets import multiclassdataset as MCD
import numpy as np

if "__main__" == __name__:
    for i in range(500):
        testc = MCD.MulticlassMOT17Eval(mode='query')
        del testc
