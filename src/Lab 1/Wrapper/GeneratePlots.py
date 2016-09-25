try:
   import cPickle as pickle
except:
   import pickle
from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np

from WrapperShared import Variant
from terminaltables import AsciiTable
import matplotlib2tikz as tikz
from scipy.interpolate import interp1d

def GeneratePlot(results, job_title, output_dir = '.'):
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
        title = "OpenMP vs Sequential (threads=" + results[0]['threads'] + ")"
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
    elif job_title == "Part C Task 1":
        xfield = "data_size"
        yfield= "relative_improvement"
        xlabel = "Matrix/Vector Size"
        ylabel = "Speedup"
        title = "OpenCL vs Sequential (x=64)"
    elif job_title == "Part C Task 2":
        xfield = "data_size"
        yfield= "relative_improvement"
        xlabel = "Matrix/Vector Size"
        ylabel = "Speedup"
        title = "OpenCL vs Sequential (x=1024)"
    else:
        return 'ERROR: job_title unknown'

    # Generate graph
    graphX = []
    graphY = []

    for result in results:
        graphX.append(result[xfield])
        graphY.append(result[yfield])

    x_sm = np.array(graphX)
    y_sm = np.array(graphY)
    graphX_smooth = np.linspace(x_sm.min(), x_sm.max(), len(graphY)*10)
    f = interp1d(graphX, graphY, kind='cubic')
    graphY_smooth = f(graphX_smooth);

    figure = plt.figure()
    style.use('ggplot')
    plt.plot(graphX_smooth,graphY_smooth)
    plt.xlim([x_sm.min(),x_sm.max()])
    plt.ylim([y_sm.min()-(y_sm.max()-y_sm.min())*0.01,y_sm.max()+(y_sm.max()-y_sm.min())*0.01])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    # convert graph to tikz
    figure.savefig('{0}/{1}.pdf'.format(output_dir,title),format='pdf',transparent=True)
    tikz.save('{0}/{1}.tikz'.format(output_dir,title),figure=figure,figureheight = '\\figureheight',figurewidth = '\\figurewidth',show_info=False,encoding='utf-8',draw_rectangles=True)