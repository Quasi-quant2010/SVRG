# -*- coding: utf-8 -*- 

import sys, gzip, codecs, optparse, os
from collections import defaultdict

import numpy as np
import pandas as pd
import cPickle, json

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
#import seaborn as sns


if __name__ == '__main__':

    parser = optparse.OptionParser()
    parser.add_option("--data", dest="data", type=str, default="adagrad_iters.txt")
    parser.add_option("--graph", dest="graph", type=str, default="result")
    (options, args) = parser.parse_args()


    plt.ylabel("LogLoss", fontsize=15)
    plt.xlabel("Iteration", fontsize=15)
    plt.minorticks_on()
    plt.grid(True)
    for idx, fname in enumerate(["Batch_L2.dat", "SGD_L2.dat", "SVRG_L2.dat"]):
        try:
            _fname = "%s/%s" % (options.data, fname)
            df = pd.read_csv(_fname, names=['iteration', fname.split(".")[0]], delimiter="\t")
            plt.plot(df["iteration"], df[fname.split(".")[0]], marker=idx+6, label=fname.split(".")[0])
        except:
            pass
    #plt.yscale('log')
    plt.legend(loc="best")
    plt.savefig("%s/%s" % (options.graph, "loss.png"))
    plt.close()


    plt.ylabel("Convergence2Optimal", fontsize=15)
    plt.xlabel("Iteration", fontsize=15)
    plt.minorticks_on()
    plt.grid(True)
    _fname = "%s/%s" % (options.data, "Batch_L2.dat")
    df_optimal = pd.read_csv(_fname, names=['iteration', 'Batch_L2.dat'.split('.')[0]], delimiter="\t")
    optimal = df_optimal.tail(1).Batch_L2.values.item()
    for idx, fname in enumerate(["SGD_L2.dat", "SVRG_L2.dat"]):
        try:
            _fname = "%s/%s" % (options.data, fname)
            df = pd.read_csv(_fname, names=['iteration', fname.split(".")[0]], delimiter="\t")
            df[fname.split(".")[0]] = df[fname.split(".")[0]] - optimal
            plt.plot(df["iteration"], df[fname.split(".")[0]], marker=idx+6, label=fname.split(".")[0])
        except:
            pass
    plt.yscale('log')
    plt.legend(loc="best")
    plt.savefig("%s/%s" % (options.graph, "convergence.png"))
    plt.close()
