import numpy as np
import os
import argparse as ap
from ExecuteBenchmarks import *
from GeneratePlots import GeneratePlot

import future        # pip install future
import builtins      # pip install future
import past          # pip install future
import six           # pip install six

iterations = 1
max_n = 5 # Times to run program to get average

make = "make all"
make_flags = ""
bin = "make schedule_single"
cdw = "../"
output_dir = "../run_output"
image_path = "images"

results = []

generate_data = True;
generate_plots = False;

def ExecuteJob(job_title,filename,iterations, images,max_n,generate_data=True,generate_plots=False):
    if generate_data:
        if PrepareBenchmark(make,make_flags,cdw) == 0:
            results = ExecuteBenchmark(job_title, iterations, images,max_n,bin,cdw, output_dir);

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
    #parser.add_argument('--output-dir', action="store", help='Output directory',default=".")
    try:
        opts = parser.parse_args(sys.argv[1:])

        #images = [f for f in os.listdir(os.path.join(cdw,image_path)) if os.path.isfile(os.path.join(cdw, image_path, f))]
        images = ['image04.bmp','image09.bmp','image15.jpg']

        ### Part A
        #output_dir = os.path.join(output_dir_root,'out')
        #if not os.path.exists(output_dir):
        #    os.makedirs(output_dir)

        ## Task 1
        ExecuteJob('main-run','main-run',iterations,images,max_n,generate_data=not opts.disable_bench,generate_plots=not opts.disable_plot)

        print("Done.")
    except SystemExit:
        print('Bad Arguments')