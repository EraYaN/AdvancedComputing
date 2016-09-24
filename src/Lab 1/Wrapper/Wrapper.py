import numpy as np
import os
import argparse as ap
from WrapperShared import Variant
from ExecuteBenchmarks import *
from GeneratePlots import GeneratePlot

# Program Definitions
OpenMP = {
    'name':'OpenMP',
    'variants':[Variant.base],
    'configs':['Release'],
    'data_sizes':[512,1024],
    'thread_range':[8]
}
OpenMPMatrix = {
    'name':'OpenMPMatrix',
    'variants':[Variant.base],
    'configs':['Release'],
    'data_sizes':[512],
    'thread_range':[8]
}
SSE = {
    'name':'SSE',
    'variants':[Variant.base],
    'configs':['Release'],
    'data_sizes':[512,1024],
    'thread_range':[8]
}
SSEMatrix = {
    'name':'SSEMatrix',
    'variants':[Variant.arbitrarysize],
    'configs':['Release','ReleaseDP'],
    'data_sizes':[512],
    'thread_range':[8]
}
AVXMatrix = {
    'name':'AVXMatrix',
    'variants':[Variant.arbitrarysize],
    'configs':['Release','ReleaseDP'],
    'data_sizes':[512],
    'thread_range':[8]
}
OpenCL = {
    'name':'OpenCL',
    'variants':[Variant.base],
    'configs':['Release'],
    'data_sizes':[512,1024],
    'thread_range':[8]
}
OpenCLMatrix = {
    'name':'OpenCLMatrix',
    'variants':[Variant.base],
    'configs':['Release','ReleaseDP'],
    'data_sizes':[512],
    'thread_range':[8]
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

def ExecuteJob(job_title,filename,platforms,types,iteration_range,max_n,generate_data=True,generate_plots=False,output_dir='.'):
    if generate_data:
        results = ExecuteBenchmark(platforms,types,iteration_range,max_n);
        PrintResults(results);
        SaveResults(filename,results)
    if generate_plots:
        if not generate_data:
            results = LoadResults(filename)
        GeneratePlot(results,job_title,output_dir)



if __name__ == '__main__':
    parser = ap.ArgumentParser(prog='ACSLabWrapper',description='ACS Benchmark Wrapper Script')
    parser.add_argument('--disable-bench', action="store_true", help='Disable the benchmarks')
    parser.add_argument('--disable-plot', action="store_true", help='Disable the plotting')
    parser.add_argument('--output-dir', action="store", help='Output directory',default="./out")
    try:
        opts = parser.parse_args(sys.argv[1:])
        output_dir = opts.output_dir;
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        SSE['data_sizes'] = np.arange(4, 1024, 4);
        ExecuteJob('Part B Task 1','partB_task1.pickle',platforms,types,iteration_range,max_n,generate_data=not opts.disable_bench,generate_plots=not opts.disable_plot,output_dir=output_dir)
        SSE['variants'] = [Variant.arbitrarysize];
        SSE['data_sizes'] = np.arange(4, 1024, 1);
        ExecuteJob('Part B Task 2','partB_task2.pickle',platforms,types,iteration_range,max_n,generate_data=not opts.disable_bench,generate_plots=not opts.disable_plot,output_dir=output_dir)
        SSE['configs'] = ['ReleaseDP'];
        ExecuteJob('Part B Task 3','partB_task3.pickle',platforms,types,iteration_range,max_n,generate_data=not opts.disable_bench,generate_plots=not opts.disable_plot,output_dir=output_dir)

        print("Done.")
    except SystemExit:
        print('Bad Arguments')