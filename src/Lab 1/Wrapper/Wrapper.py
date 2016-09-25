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
    OpenMP,
    #OpenMPMatrix,
    #SSE,
    #SSEMatrix,
    #AVXMatrix,
    #OpenCL,
    #OpenCLMatrix
]
iteration_range = [10] # range(1,11) # 1 to 10

results = []

generate_data = True;
generate_plots = False;

def ExecuteJob(job_title,filename,platforms,types,iteration_range,max_n,generate_data=True,generate_plots=False,output_dir='.'):
    if generate_data:
        results = ExecuteBenchmark(job_title, platforms,types,iteration_range,max_n);

        # Part A Task 2 needs to use the #threads with the highest speedup
        if job_title == 'openmp-number-of-threads-sweep':
            highestOpenMPSpeedup = 0
            for result in results:
                if highestOpenMPSpeedup < result['relative_improvement']:
                    OpenMP['thread_range'] = [result['threads']]
                    highestOpenMPSpeedup = result['relative_improvement']

        PrintResults(results);
        SaveResults("{0}.pickle".format(filename),results)
    if generate_plots:
        if not generate_data:
            results = LoadResults("{0}.pickle".format(filename))
        GeneratePlot(results,job_title,output_dir)



if __name__ == '__main__':
    parser = ap.ArgumentParser(prog='ACSLabWrapper',description='ACS Benchmark Wrapper Script')
    parser.add_argument('--disable-bench', action="store_true", help='Disable the benchmarks')
    parser.add_argument('--disable-plot', action="store_true", help='Disable the plotting')
    parser.add_argument('--output-dir', action="store", help='Output directory',default="../../../docs/lab1")
    try:
        opts = parser.parse_args(sys.argv[1:])
        output_dir_root = opts.output_dir;

        ### Part A
        types = [OpenMP]
        output_dir = os.path.join(output_dir_root,'OpenMP/resources')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        OpenMP['variants'] = [Variant.base]
        OpenMP['configs'] = ['Release']

        ## Task 1
        OpenMP['data_sizes'] = [2048]
        OpenMP['thread_range'] = np.arange(1, 64, 1)
        ExecuteJob('openmp-number-of-threads-sweep','openmp-number-of-threads-sweep',platforms,types,iteration_range,max_n,generate_data=not opts.disable_bench,generate_plots=not opts.disable_plot,output_dir=output_dir)

        ## Task 2
        OpenMP['data_sizes'] = np.arange(10, 10010, 100)
        # use threads from previous measurement, see ExecuteJob
        ExecuteJob('openmp-data-size-sweep','openmp-data-size-sweep',platforms,types,iteration_range,max_n,generate_data=not opts.disable_bench,generate_plots=not opts.disable_plot,output_dir=output_dir)

        ### Part B
        types = [SSE]
        output_dir = os.path.join(output_dir_root,'SSE-AVX/resources')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        SSE['variants'] = [Variant.base]
        SSE['configs'] = ['Release']
        SSE['thread_range'] = [8]

        ## Task 1
        SSE['data_sizes'] = np.arange(4, 1024, 4);
        ExecuteJob('sse-data-size-sweep','sse-data-size-sweep',platforms,types,iteration_range,max_n,generate_data=not opts.disable_bench,generate_plots=not opts.disable_plot,output_dir=output_dir)

        ## Task 2
        SSE['variants'] = [Variant.arbitrarysize];
        SSE['data_sizes'] = np.arange(4, 1024, 1);
        ExecuteJob('sse-data-size-sweep-arbitrary','sse-data-size-sweep-arbitrary',platforms,types,iteration_range,max_n,generate_data=not opts.disable_bench,generate_plots=not opts.disable_plot,output_dir=output_dir)

        ## Task 3
        SSE['configs'] = ['ReleaseDP'];
        ExecuteJob('sse-dp','sse-dp',platforms,types,iteration_range,max_n,generate_data=not opts.disable_bench,generate_plots=not opts.disable_plot,output_dir=output_dir)

        ### Part C
        types = [OpenCL]
        output_dir = os.path.join(output_dir_root,'OpenCL/resources')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        OpenCL['variants'] = [Variant.base]
        OpenCL['configs'] = ['Release']

        ## Task 1
        OpenCL['thread_range'] = [64]
        OpenCL['data_sizes'] = np.arange(64, 4096, 64)
        ExecuteJob('opencl-data-size-sweep','opencl-data-size-sweep',platforms,types,iteration_range,max_n,generate_data=not opts.disable_bench,generate_plots=not opts.disable_plot,output_dir=output_dir)

        ## Task 2
        OpenCL['thread_range'] = [2,4,8,16,32,64,128,256,512,1024]
        OpenCL['data_sizes'] = [4096]
        ExecuteJob('opencl-localsize-sweep','opencl-localsize-sweep',platforms,types,iteration_range,max_n,generate_data=not opts.disable_bench,generate_plots=not opts.disable_plot,output_dir=output_dir)


        print("Done.")
    except SystemExit:
        print('Bad Arguments')