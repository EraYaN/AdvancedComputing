print("Starting...")
import numpy as np
import os
import argparse as ap
from ExecuteBenchmarks import *


import future        # pip install future
import builtins      # pip install future
import past          # pip install future
import six           # pip install six

iterations = 1
max_n = 10 # Times to run program to get average

make = "make all"
make_flags = ""
bin = "make schedule_single"
cdw = "../"
output_dir = "../run_output"
docs_output_dir = "../../../docs/lab2"
image_path = "images"

results = []

generate_data = True;
generate_plots = False;

def ExecuteJob(job_title,filename,iterations, images,max_n,shared,generate_data=True,generate_plots=False):
    if generate_data:
        if PrepareBenchmark(make,make_flags,cdw) == 0:
            results = ExecuteBenchmark(job_title, iterations, images,max_n,shared,bin,cdw, output_dir);

            PrintResults(results);
            print("Saving pickle...")
            SaveResults("{0}.pickle".format(filename),results)

    if generate_plots:
        if not generate_data:
            print("Loading pickle...")
            results = LoadResults("{0}.pickle".format(filename))
            PrintResults(results);
        GeneratePlot(results,job_title,docs_output_dir)



if __name__ == '__main__':
    parser = ap.ArgumentParser(prog='ACSLabWrapper',description='ACS Benchmark Wrapper Script')
    parser.add_argument('--disable-bench', action="store_true", help='Disable the benchmarks')
    parser.add_argument('--disable-plot', action="store_true", help='Disable the plotting')
    #TODO report making
    #parser.add_argument('--output-dir', action="store", help='Output directory',default="../../../docs/lab2")
    #parser.add_argument('--output-dir', action="store", help='Output directory',default=".")
    try:
        opts = parser.parse_args(sys.argv[1:])
        if not opts.disable_plot:
            from GeneratePlots import GeneratePlot

        
        shared = False
        ExecuteJob('normal','main-run-normal',iterations,images,max_n,shared,generate_data=not opts.disable_bench,generate_plots=not opts.disable_plot)
        shared = True
        ExecuteJob('shared','main-run-shared',iterations,images,max_n,shared,generate_data=not opts.disable_bench,generate_plots=not opts.disable_plot)

        print("Done.")
    except SystemExit:
        print('Bad Arguments')