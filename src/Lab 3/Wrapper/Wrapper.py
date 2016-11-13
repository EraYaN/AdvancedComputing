import numpy as np
import os
import argparse as ap
from ExecuteBenchmarks import *

import colorama
colorama.init(autoreset=True)

C_RED = colorama.Fore.RED+colorama.Style.BRIGHT
C_GREEN = colorama.Fore.GREEN+colorama.Style.BRIGHT
C_CYAN = colorama.Fore.CYAN+colorama.Style.BRIGHT

import future        # pip install future
import builtins      # pip install future
import past          # pip install future
import six           # pip install six

print(C_GREEN+"Starting...")

iterations = 1
max_n = 5 # Times to run program to get average
network_sizes = [256,512]

make = "make all"
make_flags = ""
bin = "make schedule"
cdw = "../"
output_dir = "../run_output"
docs_output_dir = "../../../docs/lab3"

results = []

generate_data = True;
generate_plots = False;

def ExecuteJob(job_title,filename,iterations,max_n,network_sizes,generate_data=True,generate_plots=False):
    if generate_data:
        if PrepareBenchmark(make,make_flags,cdw) == 0:
            results = ExecuteBenchmark(job_title, iterations,max_n,network_sizes,bin,cdw, output_dir);

            print(C_CYAN+"Saving pickle...")
            SaveResults("{0}.pickle".format(filename),results)

            PrintResults(results);
            

    if generate_plots:
        if not generate_data:
            print(C_CYAN+"Loading pickle...")
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

        ExecuteJob('stage3','stage3-big-sizes',iterations,max_n,network_sizes,generate_data=not opts.disable_bench,generate_plots=not opts.disable_plot)

        print(C_GREEN+"Done.")
    except SystemExit:
        print(C_RED+'Bad Arguments')