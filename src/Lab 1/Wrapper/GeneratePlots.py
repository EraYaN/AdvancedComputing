try:
   import cPickle as pickle
except:
   import pickle
import matplotlib.pyplot as plt
import numpy as np

from WrapperShared import Variant
from terminaltables import AsciiTable
from operator import itemgetter

def GeneratePlot(results, job_title):
    # Task specific graph settings
    if job_title == "Part A Task 1":
        xfield = "threads"
        yfield= "relative_improvement"
        xlabel = "Threads"
        ylabel = "Speedup"
        title = "OpenMP vs Sequential (n=2048)"
    elif job_title == "Part A Task 2":
        xfield = "data_size"
        yfield= "relative_improvement"
        xlabel = "Data Size"
        ylabel = "Speedup"
        title = "OpenMP vs Sequential (threads=40)"
    elif job_title == "Part B Task 1":
        xfield = "data_size"
        yfield= "relative_improvement"
        xlabel = "Matrix/Vector Size"
        ylabel = "Speedup"
        title = "SSE vs Sequential"
    elif job_title == "Part B Task 2":
        xfield = "data_size"
        yfield= "relative_improvement"
        xlabel = "Matrix/Vector Size"
        ylabel = "Speedup"
        title = "SSE vs Sequential Arbitrary Size"
    elif job_title == "Part B Task 3":
        xfield = "data_size"
        yfield= "relative_improvement"
        xlabel = "Matrix/Vector Size"
        ylabel = "Speedup"
        title = "SSE vs Sequential Double Precision"

    # Generate graph
    graphX = []
    graphY = []

    for result in results:
        graphX.append(result[xfield])
        graphY.append(result[yfield])

    plt.bar(graphX,graphY,align='center')
    plt.ylim([np.min(graphY)-(np.max(graphY)-np.min(graphY))*0.01,np.max(graphY)+(np.max(graphY)-np.min(graphY))*0.01])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(title + ".png")