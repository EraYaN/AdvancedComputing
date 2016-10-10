import subprocess
import sys
import os
#import numpy as np
try:
   import cPickle as pickle
except:
   import pickle

#from terminaltables import AsciiTable
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

def ExecuteBenchmark(job_title, iterations, images, max_n=1, bin="make schedule",cwd="../",output_dir="../run_output"):
    error_occured = False
    results = []
    display_results = []
    number_of_runs = GetNumberOfRuns(iterations,images,max_n)
    current_run = 0

    for image in images:
        for iteration in range(0,iterations):
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
                result = subprocess.call("{0} RUN_IMAGE=\"{1}\" RUN_ID=\"{2}\"".format(bin,image,run_id),cwd=cwd,shell=True)
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
                        times[testName] = time_template

                    if isCuda == 1:
                        index = 'cuda'

                    times[testName][index]['preprocessing_time'] += float(preprocessing_time) / max_n
                    times[testName][index]['init_time'] += float(init_time) / max_n
                    times[testName][index]['kernel_time'] += float(kernel_time) / max_n
                    times[testName][index]['cleanup_time'] += float(cleanup_time) / max_n
                    times[testName][index]['postprocessing_time'] += float(postprocessing_time) / max_n
                    times[testName][index]['total_time'] += float(total_time) / max_n


                #if times['cuda']['total_time'] != 0:
                #    new_time += times['seq']['total_time'] /
                #    times['cuda']['total_time']

                sys.stdout.write("{3}: Run {4} out of {5}: {0: >2} out of {1: >2} ({2: >3,.0%})\r".format(n + 1, max_n,(n + 1 + current_run * max_n) / (max_n * number_of_runs['runs']),job_title,current_run,number_of_runs['runs']))
                sys.stdout.flush()


            #TODO write file.
            results.append({
                'image':image,
                "iterations":iteration,
                "times":times
                })

            #print("{0}: Run {1} out of {2} is done.
            #\r".format(job_title,current_run,number_of_runs['runs']))

    return results

def PrintResults(results):
    return
    #table_heading = ['image',
    #    'iterations',
    #    'times']
    #table_justify = {
    #    0:'left',
    #    1:'left',
    #    2:'left'
    #}
    #display_results = []
    #display_results.append(table_heading)
    #for result in results:
    #    error_text = "Yes" if result["had_error"] else "No"
    #    display_results.append([result['platform'],
    #    result['type']['name'],
    #    result['config'],
    #    result['threads'],
    #    result["data_size"],
    #    result["iterations"],
    #    result["variant"].name,
    #    "{0:.5f} us".format(result["sequential_time"] * 1000000 / result["iterations"]),
    #    "{0:.5f} us".format(result["variant_time"] * 1000000 / result["iterations"]),
    #    "{0:.3%}".format(result["relative_improvement"]),
    #    error_text])

    #results_table = AsciiTable(display_results)
    #results_table.justify_columns = table_justify
    #print(results_table.table)

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