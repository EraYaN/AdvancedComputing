import subprocess
import sys
try:
   import cPickle as pickle
except:
   import pickle

from terminaltables import AsciiTable
from WrapperShared import Variant 

# Program Definitions
OpenMP = {
    'name':'OpenMP',
    'variants':[Variant.base],
    'configs':['Release','ReleaseDP']
}
OpenMPMatrix = {
    'name':'OpenMPMatrix',
    'variants':[Variant.base],
    'configs':['Release','ReleaseDP']
}
SSE = {
    'name':'SSE',
    'variants':[Variant.arbitrarysize],
    'configs':['Release','ReleaseDP']
}
SSEMatrix = {
    'name':'SSEMatrix',
    'variants':[Variant.arbitrarysize],
    'configs':['Release','ReleaseDP','ReleaseDP']
}
OpenCL = {
    'name':'OpenCL',
    'variants':[Variant.base],
    'configs':['Release','ReleaseDP']
}
OpenCLMatrix = {
    'name':'OpenCLMatrix',
    'variants':[Variant.base],
    'configs':['Release','ReleaseDP']
}

max_n = 10 # Times to run program to get average
platforms = ['x64'] # Platform names
platform_paths = {'x86':'','x64':'x64/'} # Platform to Directory Dictionary
# Program definition array
types = [OpenMP,
    #OpenMPMatrix,
    SSE,
    #SSEMatrix,
    OpenCL,
    #OpenCLMatrix
]
thread_range = [8] # range(1,11) # 1 to 10
iteration_range = [50] # range(1,11) # 1 to 10
data_sizes = [512]

error_occured = False

results = []
display_results = []

for platform in platforms:
    for type in types:
        for config in type['configs']:
            for threads in thread_range:
                for data_size in data_sizes:
                    for iterations in iteration_range:
                        for variant in type['variants']:
                            sequential_time = 0
                            variant_time = 0
                            new_time = 0

                            for n in range(0,max_n):
                                #print("../{0}{1}/{2}.exe".format(platform_paths[platform],config,type['name']))
                                result = subprocess.run(["../{0}{1}/{2}.exe".format(platform_paths[platform],config,type['name']),"-t {0}".format(threads),"-s {0}".format(data_size),"-n {0}".format(iterations),"-v {0}".format(variant.value),'-d'],stdout=subprocess.PIPE,universal_newlines=True,cwd="../{0}{1}/".format(platform_paths[platform],config))
                                #print(result.args)
                                if result.returncode != 0:
                                    print("ERROR {1} returned {0}. Output below.".format(result.returncode, "../{0}{1}/{2}.exe".format(platform_paths[platform],config,type['name'])))
                                    print(result.stdout)
                                    error_occured = True
                                    break

                                for line in result.stdout.splitlines(): #read and store result in log file
                                    line_type = line[0:3]
                                    if line_type == "SEQ":
                                        sequential_time += float(line[4:])
                                    elif line_type == "VAR":
                                        variant_time += float(line[4:])

                                if variant_time != 0:
                                    new_time += sequential_time / variant_time

                                sys.stdout.write("{0: >2} out of {1: >2} ({2: >3,.0%})\r".format(n + 1, max_n,(n + 1) / max_n))
                                sys.stdout.flush()

                            if error_occured:
                                break
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
                                "relative_improvement":relative_improvement
                                })
                            print("Run is done.\n")
                            #print("Executed {0: <20} {1: <13} size {2: >8} /w {3:
                            #>2} thr, seq: {4: >12,.3e}, var: {5: >12,.3e}, new
                            #time {6: >12,.3%}".format("{0}
                            #{1}".format(type['name'],variant.name),"{0}
                            #{1}".format(platform,config),data_size,threads,sequential_time,variant_time,new_time))
                if error_occured:
                    break
            if error_occured:
                break
        if error_occured:
            break
    if error_occured:
        break

table_heading = [
    'platform',
    'type',
    'config',
    'threads',
    "data_size",
    "iterations",
    "variant",
    "sequential_time",
    "variant_time",
    "relative_improvement"
]
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
    9:"right"
}

display_results.append(table_heading)
for result in results:
    display_results.append([
    result['platform'],
    result['type']['name'],
    result['config'],
    result['threads'],
    result["data_size"],
    result["iterations"],
    result["variant"].name,
    "{0:.3e}".format(result["sequential_time"]),
    "{0:.3e}".format(result["variant_time"]),
    "{0:.3%}".format(result["relative_improvement"])
])

results_table = AsciiTable(display_results)
results_table.justify_columns=table_justify;
print(results_table.table)

# store data in file for later use
with open('results.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #TODO if not error process data and save figures.
if error_occured:
    print("Exited upon error.")
else:
    print("Done.");