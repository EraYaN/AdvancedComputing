import subprocess
import sys
import os
import copy
#import numpy as np
try:
   import cPickle as pickle
except:
   import pickle

from terminaltables import AsciiTable
import csv

import colorama
colorama.init(autoreset=True)

C_RED = colorama.Fore.RED+colorama.Style.BRIGHT
C_GREEN = colorama.Fore.GREEN+colorama.Style.BRIGHT
C_CYAN = colorama.Fore.CYAN+colorama.Style.BRIGHT

import future        # pip install future
import builtins      # pip install future
import past          # pip install future
import six           # pip install six

# Exit codes.  See ACSLabSharedLibrary/interactive_tools.h
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_BADARGUMENT = 2
EXIT_WRONGVALUE = 3
EXIT_OPENCLERROR = 4
EXIT_MEMORYERROR = 5
EXIT_CUDAERROR = 6
EXIT_BADINPUT = 7
EXIT_SAVEERROR = 8

LINE_MARKER = '@'
CSV_SEPARATOR = ','

def GetNumberOfRuns(iterations, max_n,network_sizes):
    result = {"runs": 0, "process_runs":0}

    for network_size in network_sizes:
        for iteration in range(0,iterations):
            for n in range(0,max_n):
                result['process_runs']+=1
            result['runs']+=1

    return result


def PrepareBenchmark(make="make all", make_flags="",cwd="../"):
    #result = subprocess.call("{0} {1}".format(make,make_flags),cwd=cwd,shell=True)
    #Build needs to be done each network size.
    result = 0
    return result

def CleanBenchmark(make="make clean", make_flags="",cwd="../"):
    result = subprocess.call("{0} {1}".format(make,make_flags),cwd=cwd,shell=True)
    return result

def ExecuteBenchmark(job_title, iterations, max_n=1,network_sizes=[64], bin="make schedule",cwd="../",output_dir="../run_output"):
    error_occured = False
    results = []
    display_results = []
    number_of_runs = GetNumberOfRuns(iterations,max_n,network_sizes)
    current_run = 0

    for network_size in network_sizes:
        CleanBenchmark()
        for iteration in range(0,iterations):
            arguments = ''
            time_template = {
                "init_time":0.0,
                "kernel1_time":0.0,
                "kernel2_time":0.0,
                "cleanup_time":0.0,
                "total_time":0.0
                }
            times = {}
            error_occured = False
            current_run+=1

            for n in range(0,max_n):
                #print("../{0}{1}/{2}.exe".format(platform_paths[platform],config,type['name']))
                run_id = "{0}-{3}x{3}-{1}-{2}".format(job_title,iteration,n,network_size)
                #print("{0} RUN_ID=\"{1}\" PROFILING=no BENCH_ARGUMENTS=\"{2}\" NETWORK_SIZE={3}".format(bin,run_id,arguments,network_size))
                #continue;
                #result = EXIT_SUCCESS
                result = subprocess.call("{0} RUN_ID=\"{1}\" PROFILING=no BENCH_ARGUMENTS=\"{2}\" NETWORK_SIZE={3}".format(bin,run_id,arguments,network_size),cwd=cwd,shell=True)
                #print(result.args)
                if result != EXIT_SUCCESS:
                    print(C_RED+"ERROR {1} returned {0}.".format(result, bin))
                    break;

                csv_data = ''
                with open(os.path.join(output_dir,run_id+'_CPU','stdout.log')) as stdout:
                    for line in stdout: #read and store result in log file
                        if line[0:1] == LINE_MARKER:
                            csv_data+=line[1:]

                with open(os.path.join(output_dir,run_id+'_CUDA','stdout.log')) as stdout:
                    for line in stdout: #read and store result in log file
                        if line[0:1] == LINE_MARKER:
                            csv_data+=line[1:]

                with open(os.path.join(output_dir,run_id+'_OPENCL','stdout.log')) as stdout:
                    for line in stdout: #read and store result in log file
                        if line[0:1] == LINE_MARKER:
                            csv_data+=line[1:]

                reader = csv.reader(csv_data.splitlines())
                #cout << LINE_MARKER << "CPU" << CSV_SEPARATOR << tInit / 1e9 << CSV_SEPARATOR << tNeighbour / 1e9 << CSV_SEPARATOR << tCompute / 1e9 << CSV_SEPARATOR << tWriteFile / 1e9 << CSV_SEPARATOR << (tInit + tLoop) / 1e9 << endl;
                for testName,init_time,kernel1_time,kernel2_time,cleanup_time,total_time in reader:

                    if testName not in times:
                        times[testName] = copy.deepcopy(time_template)

                    print(C_CYAN+"Got time for {0} {1:.3f}".format(testName, float(total_time)));

                    times[testName]['init_time'] += float(init_time) / max_n
                    times[testName]['kernel1_time'] += float(kernel1_time) / max_n
                    times[testName]['kernel2_time'] += float(kernel2_time) / max_n
                    times[testName]['cleanup_time'] += float(cleanup_time) / max_n
                    times[testName]['total_time'] += float(total_time) / max_n

                       

                #if times['cuda']['total_time'] != 0:
                #    new_time += times['seq']['total_time'] /
                #    times['cuda']['total_time']

                print(C_GREEN+"{3}: Run {4} of {5}; Iteration: {0: >2} of {1: >2} ({2: >3,.0%})\n".format(n + 1, max_n,(n + 1 + current_run * max_n) / (max_n * number_of_runs['runs']),job_title,current_run,number_of_runs['runs']))
                #sys.stdout.write("{3}: Run {4} out of {5}: {0: >2} out of {1: >2} ({2: >3,.0%})\n".format(n + 1, max_n,(n + 1 + current_run * max_n) / (max_n * number_of_runs['runs']),job_title,current_run,number_of_runs['runs']))
                #sys.stdout.flush()

            #For if the benchmarks timed out
            if 'CPU' not in times:
                times['CPU'] = copy.deepcopy(time_template)
            if 'CUDA' not in times:
                times['CUDA'] = copy.deepcopy(time_template)
            if 'OpenCL' not in times:
                times['OpenCL'] = copy.deepcopy(time_template)

            #TODO write file.
            results.append({
                "network_size":network_size,
                "iteration":iteration,
                "passes":max_n,
                "times":times,
                #"prof":prof,
                'had_error':error_occured
                })

            #print("{0}: Run {1} out of {2} is done.
            #\r".format(job_title,current_run,number_of_runs['runs']))

    return results

def GetDRAMThroughput(prof,kind='avg'):
    write = float(prof['dram_write_throughput'][kind][:4])
    read = float(prof['dram_read_throughput'][kind][:4])

    if prof['dram_read_throughput']['max'][-4:-2] == "MB":
        read = read / 1024

    if prof['dram_write_throughput']['max'][-4:-2] == "MB":
        write = write / 1024

    return write+read # GB/s

def PrintResults(results):
    table_heading = [
        'network_size',
        'iteration',
        'passes',
        'has error',
        'total-times-cpu',
        'total-times-cuda',
        'total-times-opencl',
        'kernel-times-cpu',
        'kernel-times-cuda',
        'kernel-times-opencl',
        #'peak-gflops-cuda',
        #'peak-gflops-opencl',
        #'global-dram-bandwith-max',
        #'global-dram-bandwith-min'
        ]
    table_justify = {
        0:'left',
        1:'left',
        2:'left',
        3:'left',
        4:'right',
        5:'right',
        6:'right',
        7:'right',
        #9:'right',
        #10:'right',
       # 11:'right',
        #12:'right'
    }
    display_results = []
    display_results.append(table_heading)
    for result in results:
        if "had_error" in result:
            error_text = "Yes" if result["had_error"] else "No"
        else:
            error_text = "??"

        #total_gflops = (int(result['prof']['Compute']['flop_count_sp']['max'])+int(result['prof']['Neighbour']['flop_count_sp']['max']))/1e9
        total_time_cpu = (result['times']['CPU']['total_time'])
        total_time_cuda = (result['times']['CUDA']['total_time'])
        total_time_opencl = (result['times']['OpenCL']['total_time'])
        kernel_time_cpu = (result['times']['CPU']['kernel1_time']+result['times']['CPU']['kernel2_time'])
        kernel_time_cuda = (result['times']['CUDA']['kernel1_time']+result['times']['CUDA']['kernel2_time'])
        kernel_time_opencl = (result['times']['OpenCL']['kernel1_time']+result['times']['OpenCL']['kernel2_time'])

        #total_global_dram_max = max(GetDRAMThroughput(result['prof']['Compute'],'max'),
        #                                 GetDRAMThroughput(result['prof']['Neighbour'],'max'))

        #total_global_dram_min = min(GetDRAMThroughput(result['prof']['Compute'],'min'),
        #                                 GetDRAMThroughput(result['prof']['Neighbour'],'min'))

        display_results.append([
            result['network_size'],
        result['iteration']+1,
        result['passes'],
        error_text,
        "{0:.2f} s".format(total_time_cpu),
        "{0:.2f} s".format(total_time_cuda),
        "{0:.2f} s".format(total_time_opencl),
        "{0:.2f} s".format(kernel_time_cpu),
        "{0:.2f} s".format(kernel_time_cuda),
        "{0:.2f} s".format(kernel_time_opencl),
        #"{0:.2f} GFLOPS".format(total_gflops/(total_time_cuda/1000)),
        #"{0:.2f} GFLOPS".format(total_gflops/(total_time_opencl/1000)),
        #"{0:.2f} GB/s".format(total_global_dram_max),
        #"{0:.2f} GB/s".format(total_global_dram_min)
        ])

    results_table = AsciiTable(display_results)
    results_table.justify_columns = table_justify
    print(results_table.table)

def SaveResults(filename,results):
    # store data in file for later use
    with open(filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

def LoadResults(filename):
    # store data in file for later use
    with open(filename, 'rb') as handle:
        results = pickle.load(handle)
        return results
    return [];