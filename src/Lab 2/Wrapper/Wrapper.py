import numpy as np
import os
import argparse as ap
from WrapperShared import Variant
from ExecuteBenchmarks import *
from GeneratePlots import GeneratePlot

# Program Definitions
CUDA = {
    'name':'cuda',
    'variants':[Variant.base],
    'configs':['Release'],
    'images':['image00.jpg']
}
Sequential = {
    'name':'sequential',
    'variants':[Variant.base],
    'configs':['Release'],
    'images':['image00.jpg']
}

max_n = 5 # Times to run program to get average
platforms = ['x64'] # Platform names

# Program definition array
types = [
    Sequential,
    CUDA
]
#TODO implement
iteration_range = [1] # range(1,11) # 1 to 10

results = []

generate_data = True;
generate_plots = False;

def ExecuteJob(job_title,filename,platforms,types,iteration_range,max_n,generate_data=True,generate_plots=False,output_dir='.'):
    if generate_data:
        results = ExecuteBenchmark(job_title, platforms,types,iteration_range,max_n);

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
    #TODO report making
    #parser.add_argument('--output-dir', action="store", help='Output directory',default="../../../docs/lab2")
    parser.add_argument('--output-dir', action="store", help='Output directory',default=".")
    try:
        opts = parser.parse_args(sys.argv[1:])
        output_dir_root = opts.output_dir;

        ### Part A
        types = [Sequential,CUDA]
        output_dir = os.path.join(output_dir_root,'out')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        ## Task 1
        ExecuteJob('sequential-test-run','sequential-test-run',platforms,types,iteration_range,max_n,generate_data=not opts.disable_bench,generate_plots=not opts.disable_plot,output_dir=output_dir)

        print("Done.")
    except SystemExit:
        print('Bad Arguments')