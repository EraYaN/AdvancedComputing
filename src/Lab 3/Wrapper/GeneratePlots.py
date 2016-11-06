from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np

from ExecuteBenchmarks import GetDRAMThroughput
import jinja2
import os
from jinja2 import Template
latex_jinja_env = jinja2.Environment(
	block_start_string = '\BLOCK{',
	block_end_string = '}',
	variable_start_string = '\VAR{',
	variable_end_string = '}',
	comment_start_string = '\#{',
	comment_end_string = '}',
	line_statement_prefix = '%%',
	line_comment_prefix = '%#',
	trim_blocks = True,
	autoescape = False,
	loader = jinja2.FileSystemLoader(os.path.abspath('.'))
)

# GeForce 750 Ti
MAX_GFLOPS = 1398
MAX_BANDWIDTH = 86.4

systemName = "750Ti Srv."

def GeneratePlot(results, job_title, output_dir = '.'):
    print("Processing Data...")
    # Generate graph
    ind = []

    columns = {}

    for result in results:
        for testName in result['times']:
            time_data = result['times'][testName]
            if testName not in columns:
                columns[testName] = {}

            imageNumber = int(result['image'][5:7])
            if imageNumber not in  columns[testName]:
                columns[testName][imageNumber] = {}

            columns[testName][imageNumber]['total-seq-time'] = time_data['seq']['total_time']/1e6; # ms
            columns[testName][imageNumber]['total-cuda-time'] = time_data['cuda']['total_time']/1e6; # ms
            columns[testName][imageNumber]['total-speedup'] = time_data['seq']['total_time']/time_data['cuda']['total_time'];

            columns[testName][imageNumber]['kernel-seq-time'] = time_data['seq']['kernel_time']/1e6; # ms
            columns[testName][imageNumber]['kernel-cuda-time'] = time_data['cuda']['kernel_time']/1e3; # us
            columns[testName][imageNumber]['kernel-speedup'] = time_data['seq']['kernel_time']/time_data['cuda']['kernel_time'];

            #print("{0:e} GFLOPS".format(int(result['prof'][testName]['flop_count_sp']['max'])/1e9))
            #print("{0:e} seconds".format(time['cuda']['kernel_time']/1e9))
            #print("{0:f} GFLOPS/s".format((int(result['prof'][testName]['flop_count_sp']['max'])/1e9)/(time['cuda']['kernel_time']/1e9)))

            columns[testName][imageNumber]['theoretical-gflops'] = MAX_GFLOPS;
            columns[testName][imageNumber]['theoretical-bandwidth'] = MAX_BANDWIDTH;

            columns[testName][imageNumber]['total-mflops'] = (int(result['prof'][testName]['flop_count_sp']['max'])/1e6); # MFLOPS
            #columns[testName][imageNumber]['attained-gflops'] = (int(result['prof'][testName]['flop_count_sp']['max'])/1e9)/(time_data['cuda']['kernel_time']/1e9); # GFLOPS/s
            columns[testName][imageNumber]['reported-sp-efficiency'] = float(result['prof'][testName]['flop_sp_efficiency']['max'][:-1])
            columns[testName][imageNumber]['attained-gflops'] = MAX_GFLOPS*columns[testName][imageNumber]['reported-sp-efficiency']/100; # GFLOPS/s
            columns[testName][imageNumber]['attained-bandwidth'] = GetDRAMThroughput(result['prof'][testName],'max'); # GB/s

            columns[testName][imageNumber]['reported-sm-efficiency'] = float(result['prof'][testName]['sm_efficiency']['max'][:-1])
            columns[testName][imageNumber]['attained-iops'] = (int(result['prof'][testName]['inst_integer']['max'][:-1])/2e9)/(time_data['cuda']['kernel_time']/1e9); # GIOPS/s (2 because documentation says so)

    template = latex_jinja_env.get_template('table.tex')
    print("Writing Latex Tables...")
    for testName in columns:
        table_latex = template.render(systemName=systemName,test=columns[testName],testName=testName,testNameLower=testName.lower())
        resources_dir = os.path.join(output_dir,testName.lower(),'resources')
        if not os.path.exists(resources_dir):
            os.makedirs(resources_dir)
        with open(os.path.join(resources_dir,'perf-table{0}.tex'.format(job_title)),'w') as file:
            file.write(table_latex)

    print("Generating Plot...")
    xlabel = "Kernel"
    ylabel = "Speedup"
    title = "Total CUDA Speedup"
    style.use('bmh')

    ind = np.arange(4)
    width = 0.25


    figure, ax = plt.subplots()
    legend_items = []
    rects = []
    xticklabels = []
    prop_iter = iter(plt.rcParams['axes.prop_cycle'])
    i=0
    for result in results:
        imageNumber = int(result['image'][5:7])
        values = []
        for testName in result['times']:
            values.append(columns[testName][imageNumber]['total-speedup'])
            if testName not in xticklabels:
                xticklabels.append(testName)
        legend_items.append(result['image'])
        rects.append(ax.bar(ind+width*i, values, width, color=next(prop_iter)['color']))
        i += 1

    ax.legend(rects, legend_items)

    ax.set_xticklabels(xticklabels)
    ax.set_xticks(ind + width*i/2)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_yscale("log", nonposy='clip')
    ax.set_title(title)

    plt.grid(True)

    print("Saving Plot...")
    resources_dir = os.path.join(output_dir,'conclusion','resources')
    if not os.path.exists(resources_dir):
        os.makedirs(resources_dir)
    figure.savefig(os.path.join(resources_dir,"{0}-{1}.pdf".format("total-cuda-speedup",job_title)),format='pdf',transparent=True)

    print("Generating Plot...")
    xlabel = "Kernel"
    ylabel = "Speedup"
    title = "Kernel CUDA Speedup"
    style.use('bmh')

    ind = np.arange(4)
    width = 0.25


    figure, ax = plt.subplots()
    legend_items = []
    rects = []
    xticklabels = []
    prop_iter = iter(plt.rcParams['axes.prop_cycle'])
    i=0
    for result in results:
        imageNumber = int(result['image'][5:7])
        values = []
        for testName in result['times']:
            values.append(columns[testName][imageNumber]['kernel-speedup'])
            if testName not in xticklabels:
                xticklabels.append(testName)
        legend_items.append(result['image'])
        rects.append(ax.bar(ind+width*i, values, width, color=next(prop_iter)['color']))
        i += 1

    ax.legend(rects, legend_items)

    ax.set_xticklabels(xticklabels)
    ax.set_xticks(ind + width*i/2)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_yscale("log", nonposy='clip')
    ax.set_title(title)

    plt.grid(True)

    print("Saving Plot...")
    resources_dir = os.path.join(output_dir,'conclusion','resources')
    if not os.path.exists(resources_dir):
        os.makedirs(resources_dir)
    figure.savefig(os.path.join(resources_dir,"{0}-{1}.pdf".format("kernel-cuda-speedup",job_title)),format='pdf',transparent=True)