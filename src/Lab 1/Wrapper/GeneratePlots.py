try:
   import cPickle as pickle
except:
   import pickle
from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np

from WrapperShared import Variant
from terminaltables import AsciiTable
#import matplotlib2tikz as tikz
from scipy.optimize import curve_fit

def func_exp(x, a, b):
    return a*np.exp(x*b)

def func_log(x, a, b):
    return a*np.log(x*b)

def func_poly3(x, a, b, c, d):
    return a*x**3 + b*x**2 +c*x + d

def func_poly5(x, a, b, c, d, e, f):
    return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f

def GeneratePlot(results, job_title, output_dir = '.'):
    # Task specific graph settings
    if job_title == "openmp-number-of-threads-sweep":
        xfield = "threads"
        yfield= "relative_improvement"
        xlabel = "Threads"
        ylabel = "Speedup"
        title = "OpenMP vs Sequential (data_size={0})".format(results[0]['data_size'])
        order = func_poly3
    elif job_title == "openmp-data-size-sweep":
        xfield = "data_size"
        yfield= "relative_improvement"
        xlabel = "Matrix/Vector Size"
        ylabel = "Speedup"
        title = "OpenMP vs Sequential (threads={0})".format(results[0]['threads'])
        order = func_poly5
    elif job_title == "sse-data-size-sweep":
        xfield = "data_size"
        yfield= "relative_improvement"
        xlabel = "Matrix/Vector Size"
        ylabel = "Speedup"
        title = "SSE vs Sequential (threads={0})".format(results[0]['threads'])
        order = func_poly5
    elif job_title == "sse-data-size-sweep-arbitrary":
        xfield = "data_size"
        yfield= "relative_improvement"
        xlabel = "Matrix/Vector Size"
        ylabel = "Speedup"
        title = "SSE vs Sequential Arbitrary Size (threads={0})".format(results[0]['threads'])
        order = func_poly5
    elif job_title == "sse-dp":
        xfield = "data_size"
        yfield= "relative_improvement"
        xlabel = "Matrix/Vector Size"
        ylabel = "Speedup"
        title = "SSE vs Sequential Double Precision (threads={0})"
        order = func_poly5
    elif job_title == "opencl-data-size-sweep":
        xfield = "data_size"
        yfield= "relative_improvement"
        xlabel = "Matrix/Vector Size"
        ylabel = "Speedup"
        title = "OpenCL vs Sequential (localSize={0})".format(results[0]['threads'])
        order = func_poly5
    elif job_title == "opencl-localsize-sweep":
        xfield = "threads"
        yfield= "relative_improvement"
        xlabel = "Local Work Group Size"
        ylabel = "Speedup"
        title = "OpenCL vs Sequential (data_size={0})".format(results[0]['data_size'])
        order = func_log
    else:
        return 'ERROR: job_title unknown'

    print("Processing Data...")
    # Generate graph
    graphX = []
    graphY = []

    for result in results:
        graphX.append(result[xfield])
        graphY.append(result[yfield])

    #print("Fitting Curve...")
    x_sm = np.array(graphX)
    y_sm = np.array(graphY)
    #graphX_smooth = np.linspace(x_sm.min(), x_sm.max(), len(graphY)*10)
    #graphY_smooth = spline(graphX, graphY, graphX_smooth, kind = "smoothest")
    #p = np.polyfit(graphX, graphY, order)
    #f = np.poly1d(p)

    #popt, pcov = curve_fit(order, graphX, graphY)
    #graphY_smooth = order(graphX_smooth, *popt);

    print("Generating Plot...")
    figure = plt.figure()
    style.use('bmh')
    plt.scatter(graphX,graphY)
    #plt.plot(graphX_smooth,graphY_smooth)
    plt.xlim([x_sm.min()-(x_sm.max()-x_sm.min())*0.05,x_sm.max()+(x_sm.max()-x_sm.min())*0.05])
    plt.ylim([y_sm.min()-(y_sm.max()-y_sm.min())*0.05,y_sm.max()+(y_sm.max()-y_sm.min())*0.05])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    print("Saving Plot...")
    # convert graph to tikz
    figure.savefig('{0}/{1}.pdf'.format(output_dir,job_title),format='pdf',transparent=True)
    #tikz.save('{0}/{1}.tikz'.format(output_dir,job_title),figure=figure,figureheight = '\\figureheight',figurewidth = '\\figurewidth',show_info=True,encoding='utf-8',draw_rectangles=True)