import numpy as np
import argparse as ap
from WrapperShared import Variant
from ExecuteBenchmarks import *
from GeneratePlots import GeneratePlot

# Program Definitions
OpenMP = {
    'name':'OpenMP',
    'variants':[Variant.base],
    'configs':['Release'],
    'data_sizes':[2048],
    'thread_range':np.arange(1, 65, 1)
}
OpenMPMatrix = {
    'name':'OpenMPMatrix',
    'variants':[Variant.base],
    'configs':['Release'],
    'data_sizes':[512],
    'thread_range':np.arange(1, 65, 1)
}
SSE = {
    'name':'SSE',
    'variants':[Variant.base],
    'configs':['Release'],
    'data_sizes':np.arange(4, 1024, 4),
    'thread_range':[8]
}
SSEMatrix = {
    'name':'SSEMatrix',
    'variants':[Variant.arbitrarysize],
    'configs':['Release','ReleaseDP'],
    'data_sizes':[512,1024],
    'thread_range':[4,8]
}
AVXMatrix = {
    'name':'AVXMatrix',
    'variants':[Variant.arbitrarysize],
    'configs':['Release','ReleaseDP'],
    'data_sizes':[512,1024],
    'thread_range':[4,8]
}
OpenCL = {
    'name':'OpenCL',
    'variants':[Variant.base],
    'configs':['Release','ReleaseDP'],
    'data_sizes':[1024],
    'thread_range':[8,16,32,64,128]
}
OpenCLMatrix = {
    'name':'OpenCLMatrix',
    'variants':[Variant.base],
    'configs':['Release','ReleaseDP'],
    'data_sizes':[256],
    'thread_range':[2,4,8,16,32,64,128]
}

max_n = 5 # Times to run program to get average
platforms = ['x64'] # Platform names

# Program definition array
types = [
    #OpenMP,
    #OpenMPMatrix,
    SSE,
    #SSEMatrix,
    #AVXMatrix,
    #OpenCL,
    #OpenCLMatrix
]
iteration_range = [5] # range(1,11) # 1 to 10

results = []

generate_data = True;
generate_plots = False;

def ExecuteJob(job_title,filename,platforms,types,iteration_range,max_n,generate_data=True,generate_plots=False):
    if generate_data:
        results = ExecuteBenchmark(platforms,types,iteration_range,max_n);
        PrintResults(results);
        SaveResults(filename,results)
    if generate_plots:
        if not generate_data:
            results = LoadResults(filename)
        GeneratePlot(results,job_title)



if __name__ == '__main__':
    parser = ap.ArgumentParser(prog='ACSLabWrapper',description='ACS Benchmark Wrapper Script')
    parser.add_argument('--disable-bench', action="store_true", help='Disable the benchmarks')
    parser.add_argument('--disable-plot', action="store_true", help='Disable the plotting')
    try:
        opts = parser.parse_args(sys.argv[1:])

        ExecuteJob('Part B Task 1','partB_task1.pickle',platforms,types,iteration_range,max_n,generate_data=not opts.disable_bench,generate_plots=not opts.disable_plot)
        SSE['variant'] = [Variant.arbitrarysize];
        SSE['data_sizes'] = np.arange(4, 1024, 1);
        ExecuteJob('Part B Task 2','partB_task2.pickle',platforms,types,iteration_range,max_n,generate_data=not opts.disable_bench,generate_plots=not opts.disable_plot)
        SSE['configs'] = ['ReleaseDP'];
        ExecuteJob('Part B Task 3','partB_task3.pickle',platforms,types,iteration_range,max_n,generate_data=not opts.disable_bench,generate_plots=not opts.disable_plot)

        print("Done.")
    except SystemExit:
        print('Bad Arguments')