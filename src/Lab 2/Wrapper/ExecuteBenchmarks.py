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

def GetNumberOfRuns(iterations, images, max_n):
    result = {"runs": 0, "process_runs":0}

    for image in images:
        for iteration in range(0,iterations):
            for n in range(0,max_n):
                result['process_runs']+=1
            result['runs']+=1

    return result


def PrepareBenchmark(make="make all", make_flags="",cwd="../"):
    result = subprocess.call("{0} {1}".format(make,make_flags),cwd=cwd,shell=True)
    return result

def ExecuteBenchmark(job_title, iterations, images, max_n=1, shared=False, bin="make schedule",cwd="../",output_dir="../run_output"):
    error_occured = False
    results = []
    display_results = []
    number_of_runs = GetNumberOfRuns(iterations,images,max_n)
    current_run = 0

    for image in images:
        for iteration in range(0,iterations):
            run_id = "{0}-{1}-{2}".format(image,iteration,"p")
            #arguments = '--save-images'
            arguments = ''
            if shared:
                arguments += ' --shared-histogram-kernel'
            result = subprocess.call("{0} RUN_IMAGE=\"{1}\" RUN_ID=\"{2}\" PROFILING=yes BENCH_ARGUMENTS=\"{3}\"".format(bin,image,run_id,arguments),cwd=cwd,shell=True)
            if result != EXIT_SUCCESS:
                print("ERROR {1} returned {0}.".format(result, bin))
                break;

            prof = {"Smooth":{},"Grayscale":{},"Histogram":{},"Contrast":{}}
            print("Getting profiling info.")
            with open(os.path.join(output_dir,run_id,"stderr.log"),'r') as nvprof_stderr:
                reader = csv.DictReader(row for row in nvprof_stderr if not row.startswith('='))
                for prof_dict in reader:
                    kernel=''
                    if prof_dict['Kernel'].startswith("rgb2grayCuda"):
                        kernel = "Grayscale"
                    if prof_dict['Kernel'].startswith("contrast1DCuda"):
                        kernel = "Contrast"
                    if prof_dict['Kernel'].startswith("triangularSmoothCuda"):
                        kernel = "Smooth"
                    if prof_dict['Kernel'].startswith("histogram1DCuda"):
                        kernel = "Histogram"

                    if kernel in prof:
                        attribute = prof_dict['Metric Name']
                        value_min = prof_dict['Min']
                        value_avg = prof_dict['Avg']
                        value_max = prof_dict['Max']

                        if attribute not in prof[kernel]:
                            prof[kernel][attribute]={}

                        prof[kernel][attribute]['min'] = value_min
                        prof[kernel][attribute]['avg'] = value_avg
                        prof[kernel][attribute]['max'] = value_max
            print("Parsed and saved profiling info.")
            time_template = {
                "cuda":{
                    "preprocessing_time":0.0,
                    "init_time":0.0,
                    "kernel_time":0.0,
                    "cleanup_time":0.0,
                    "postprocessing_time":0.0,
                    "total_time":0.0
                    },
                "seq":{
                    "preprocessing_time":0.0,
                    "init_time":0.0,
                    "kernel_time":0.0,
                    "cleanup_time":0.0,
                    "postprocessing_time":0.0,
                    "total_time":0.0
                    }
                }
            times = {}
            error_occured = False
            current_run+=1

            for n in range(0,max_n):
                #print("../{0}{1}/{2}.exe".format(platform_paths[platform],config,type['name']))
                run_id = "{0}-{1}-{2}".format(image,iteration,n)
                result = subprocess.call("{0} RUN_IMAGE=\"{1}\" RUN_ID=\"{2}\" PROFILING=no BENCH_ARGUMENTS=\"{3}\"".format(bin,image,run_id,arguments),cwd=cwd,shell=True)
                #print(result.args)
                if result != EXIT_SUCCESS:
                    print("ERROR {1} returned {0}.".format(result, bin))
                    break;

                csv_data = ''
                with open(os.path.join(output_dir,run_id,'stdout.log')) as stdout:
                    for line in stdout: #read and store result in log file
                        if line[0:1] == LINE_MARKER:
                            csv_data+=line[1:]

                reader = csv.reader(csv_data.splitlines())
                for testName, testId, isCuda, preprocessing_time,init_time,kernel_time,cleanup_time,postprocessing_time,total_time in reader:
                    index = 'seq'
                    if testName not in times:
                        times[testName] = copy.deepcopy(time_template)

                    if int(isCuda) == 1:
                        index = 'cuda'

                    print("Got time for {0} {1} {2:.0f}".format(testName, index, float(total_time)));

                    times[testName][index]['preprocessing_time'] += float(preprocessing_time) / max_n
                    times[testName][index]['init_time'] += float(init_time) / max_n
                    times[testName][index]['kernel_time'] += float(kernel_time) / max_n
                    times[testName][index]['cleanup_time'] += float(cleanup_time) / max_n
                    times[testName][index]['postprocessing_time'] += float(postprocessing_time) / max_n
                    times[testName][index]['total_time'] += float(total_time) / max_n


                #if times['cuda']['total_time'] != 0:
                #    new_time += times['seq']['total_time'] /
                #    times['cuda']['total_time']

                sys.stdout.write("{3}: Run {4} out of {5}: {0: >2} out of {1: >2} ({2: >3,.0%})\n".format(n + 1, max_n,(n + 1 + current_run * max_n) / (max_n * number_of_runs['runs']),job_title,current_run,number_of_runs['runs']))
                sys.stdout.flush()


            #TODO write file.
            results.append({
                'image':image,
                "iteration":iteration,
                "passes":max_n,
                "times":times,
                "prof":prof,
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
    table_heading = ['image',
        'iteration',
        'passes',
        'has error',
        'total-times-seq',
        'total-times-cuda',
        'kernel-times-seq',
        'kernel-times-cuda',
        'peak-gflops',
        'global-dram-bandwith-max',
        'global-dram-bandwith-min'
        ]
    table_justify = {
        0:'left',
        1:'left',
        2:'left',
        4:'left',
        5:'right',
        6:'right',
        7:'right',
        8:'right',
        9:'right'
    }
    display_results = []
    display_results.append(table_heading)
    for result in results:
        if "had_error" in result:
            error_text = "Yes" if result["had_error"] else "No"
        else:
            error_text = "??"

        total_gflops = (int(result['prof']['Grayscale']['flop_count_sp']['max'])+int(result['prof']['Histogram']['flop_count_sp']['max'])+int(result['prof']['Contrast']['flop_count_sp']['max'])+int(result['prof']['Smooth']['flop_count_sp']['max']))/1e9
        total_time_seq = (result['times']['Grayscale']['seq']['total_time']+result['times']['Histogram']['seq']['total_time']+result['times']['Contrast']['seq']['total_time']+result['times']['Smooth']['seq']['total_time'])/1e6
        total_time_cuda = (result['times']['Grayscale']['cuda']['total_time']+result['times']['Histogram']['cuda']['total_time']+result['times']['Contrast']['cuda']['total_time']+result['times']['Smooth']['cuda']['total_time'])/1e6
        kernel_time_seq = (result['times']['Grayscale']['seq']['kernel_time']+result['times']['Histogram']['seq']['kernel_time']+result['times']['Contrast']['seq']['kernel_time']+result['times']['Smooth']['seq']['kernel_time'])/1e6
        kernel_time_cuda = (result['times']['Grayscale']['cuda']['kernel_time']+result['times']['Histogram']['cuda']['kernel_time']+result['times']['Contrast']['cuda']['kernel_time']+result['times']['Smooth']['cuda']['kernel_time'])/1e6

        total_global_dram_max = max(GetDRAMThroughput(result['prof']['Grayscale'],'max'),
                                         GetDRAMThroughput(result['prof']['Histogram'],'max'),
                                         GetDRAMThroughput(result['prof']['Contrast'],'max'),
                                         GetDRAMThroughput(result['prof']['Smooth'],'max'))

        total_global_dram_min = min(GetDRAMThroughput(result['prof']['Grayscale'],'min'),
                                         GetDRAMThroughput(result['prof']['Histogram'],'min'),
                                         GetDRAMThroughput(result['prof']['Contrast'],'min'),
                                         GetDRAMThroughput(result['prof']['Smooth'],'min'))

        display_results.append([result['image'],
        result['iteration']+1,
        result['passes'],
        error_text,
        "{0:.2f} ms".format(total_time_seq),
        "{0:.2f} ms".format(total_time_cuda),
        "{0:.2f} ms".format(kernel_time_seq),
        "{0:.2f} ms".format(kernel_time_cuda),
        "{0:.2f} GFLOPS".format(total_gflops/(total_time_cuda/1000)),
        "{0:.2f} GB/s".format(total_global_dram_max),
        "{0:.2f} GB/s".format(total_global_dram_min)
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