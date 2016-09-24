import subprocess
import sys
import numpy as np
try:
   import cPickle as pickle
except:
   import pickle

from terminaltables import AsciiTable
from WrapperShared import Variant

# Exit codes.  See ACSLabSharedLibrary/interactive_tools.h
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_BADARGUMENT = -1
EXIT_WRONGVALUE = -2
EXIT_OPENCLERROR = -3
EXIT_MEMORYERROR = -4

def GetNumberOfRuns(platforms,types,iteration_range,max_n):
    result = {"runs": 0, "process_runs":0}

    for platform in platforms:
        for type in types:
            total_runs = len(platforms)*len(types)*len(iteration_range)
            for config in type['configs']:
                for threads in type['thread_range']:
                    for data_size in type['data_sizes']:
                        for iterations in iteration_range:
                            for variant in type['variants']:
                                for n in range(0,max_n):
                                    result['process_runs']+=1
                                result['runs']+=1

    return result;


def ExecuteBenchmark(platforms,types,iteration_range,max_n=1):
    platform_paths = {'x86':'','x64':'x64/'} # Platform to Directory Dictionary
    error_occured = False
    results = []
    display_results = []
    number_of_runs = GetNumberOfRuns(platforms,types,iteration_range,max_n)
    current_run = 0
    for platform in platforms:
        for type in types:
            total_runs = len(platforms)*len(types)*len(iteration_range)
            for config in type['configs']:
                for threads in type['thread_range']:
                    for data_size in type['data_sizes']:
                        for iterations in iteration_range:
                            for variant in type['variants']:
                                sequential_time = 0
                                variant_time = 0
                                new_time = 0
                                error_occured = False

                                for n in range(0,max_n):
                                    #print("../{0}{1}/{2}.exe".format(platform_paths[platform],config,type['name']))
                                    result = subprocess.run(["../{0}{1}/{2}.exe".format(platform_paths[platform],config,type['name']),"-t {0}".format(threads),"-s {0}".format(data_size),"-n {0}".format(iterations),"-v {0}".format(variant.value)],stdout=subprocess.PIPE,universal_newlines=True,cwd="../{0}{1}/".format(platform_paths[platform],config))
                                    #print(result.args)
                                    if result.returncode != EXIT_SUCCESS:
                                        print("ERROR {1} returned {0}. Output below.".format(result.returncode, "../{0}{1}/{2}.exe".format(platform_paths[platform],config,type['name'])))
                                        print(result.stdout)
                                        error_occured = True
                                        if result.returncode != EXIT_WRONGVALUE:
                                            break

                                    for line in result.stdout.splitlines(): #read and store result in log file
                                        line_type = line[0:3]
                                        if line_type == "SEQ":
                                            sequential_time += float(line[4:])
                                        elif line_type == "VAR":
                                            variant_time += float(line[4:])

                                    if variant_time != 0:
                                        new_time += sequential_time / variant_time

                                    sys.stdout.write("{3}: {0: >2} out of {1: >2} ({2: >3,.0%})\r".format(n + 1, max_n,(n + 1) / max_n,type['name']))
                                    sys.stdout.flush()

                                sequential_time = sequential_time / max_n
                                variant_time = variant_time / max_n
                                relative_improvement = new_time / max_n

                                #TODO write file.
                                results.append({
                                    'type':type,
                                    'variant':variant,
                                    'platform':platform,
                                    'config':config,
                                    "data_size":data_size,
                                    "iterations":iterations,
                                    "threads":threads,
                                    "sequential_time":sequential_time,
                                    "variant_time":variant_time,
                                    "relative_improvement":relative_improvement,
                                    "had_error":error_occured
                                    })
                                current_run+=1
                                print("{0}: Run {1} out of {2} is done.\n".format(type['name'],current_run,number_of_runs['runs']))

    return results;

def PrintResults(results):
    table_heading = ['platform',
        'type',
        'config',
        'threads',
        "data_size",
        "iterations",
        "variant",
        "sequential_time",
        "variant_time",
        "relative_improvement",
        "had_error"]
    table_justify = {
        0:'left',
        1:'left',
        2:'left',
        3:'left',
        4:"right",
        5:"right",
        6:"left",
        7:"right",
        8:"right",
        9:"right",
        10:"left"
    }
    display_results = [];
    display_results.append(table_heading)
    for result in results:
        error_text = "Yes" if result["had_error"] else "No"
        display_results.append([result['platform'],
        result['type']['name'],
        result['config'],
        result['threads'],
        result["data_size"],
        result["iterations"],
        result["variant"].name,
        "{0:.5f} us".format(result["sequential_time"] * 1000000 / result["iterations"]),
        "{0:.5f} us".format(result["variant_time"] * 1000000 / result["iterations"]),
        "{0:.3%}".format(result["relative_improvement"]),
        error_text])

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
        return results;
    return [];