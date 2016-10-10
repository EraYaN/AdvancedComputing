#try:
#   import cPickle as pickle
#except:
#   import pickle
#from matplotlib import pyplot as plt
#from matplotlib import style
#import numpy as np

#from WrapperShared import Variant
#from terminaltables import AsciiTable
#import matplotlib2tikz as tikz
#from scipy.optimize import curve_fit


def GeneratePlot(results, job_title, output_dir = '.'):
    #TODO update plotting for difference plotting (comparison) maybe even with the diff images.
    return "NotImplemented"

    # Task specific graph settings
    #if job_title == "openmp-number-of-threads-sweep":
    #    xfield = "threads"
    #    yfield= "relative_improvement"
    #    xlabel = "Threads"
    #    ylabel = "Speedup"
    #    title = "OpenMP vs Sequential (data_size={0})".format(results[0]['data_size'])
    #    order = func_poly3
    #else:
    #    return 'ERROR: job_title unknown'

    #print("Processing Data...")
    ## Generate graph
    #graphX = []
    #graphY = []

    #for result in results:
    #    graphX.append(result[xfield])
    #    graphY.append(result[yfield])

    ##print("Fitting Curve...")
    #x_sm = np.array(graphX)
    #y_sm = np.array(graphY)
    ##graphX_smooth = np.linspace(x_sm.min(), x_sm.max(), len(graphY)*10)
    ##graphY_smooth = spline(graphX, graphY, graphX_smooth, kind = "smoothest")
    ##p = np.polyfit(graphX, graphY, order)
    ##f = np.poly1d(p)

    ##popt, pcov = curve_fit(order, graphX, graphY)
    ##graphY_smooth = order(graphX_smooth, *popt);

    #print("Generating Plot...")
    #figure = plt.figure()
    #style.use('bmh')
    #plt.scatter(graphX,graphY)
    ##plt.plot(graphX_smooth,graphY_smooth)
    #plt.xlim([x_sm.min()-(x_sm.max()-x_sm.min())*0.05,x_sm.max()+(x_sm.max()-x_sm.min())*0.05])
    #plt.ylim([y_sm.min()-(y_sm.max()-y_sm.min())*0.05,y_sm.max()+(y_sm.max()-y_sm.min())*0.05])
    #plt.xlabel(xlabel)
    #plt.ylabel(ylabel)
    #plt.title(title)
    #plt.grid(True)

    #print("Saving Plot...")
    ## convert graph to tikz
    #figure.savefig('{0}/{1}.pdf'.format(output_dir,job_title),format='pdf',transparent=True)
    #tikz.save('{0}/{1}.tikz'.format(output_dir,job_title),figure=figure,figureheight = '\\figureheight',figurewidth = '\\figurewidth',show_info=True,encoding='utf-8',draw_rectangles=True)