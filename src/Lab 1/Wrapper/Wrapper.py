import subprocess
import sys
from enum import Enum

# Global Variants Definition

class Variant(Enum):
    base = 0

# Program Definitions
OpenMP = {'name':'OpenMP','variants':[Variant.base]}
OpenMPMatrix = {'name':'OpenMPMatrix','variants':[Variant.base]}
SSE = {'name':'SSE','variants':[Variant.base]}
SSEMatrix = {'name':'SSEMatrix','variants':[Variant.base]}
OpenCL = {'name':'OpenCL','variants':[Variant.base]}
OpenCLMatrix = {'name':'OpenCLMatrix','variants':[Variant.base]}

max_n = 50 # Times to run program to get average

platforms = ['x86','x64'] # Platform names
platform_paths = {'x86':'','x64':'x64/'} # Platform to Directory Dictionary
configs = ['Release','Debug'] # Build config names
types = [OpenMP] # Program names
max_thread = 10
thread_range = [2,4,8] # range(1,max_thread+1)
data_sizes = [1000]

error_occured = False;

for platform in platforms:
    for config in configs:
        #print("Running config {0}".format(config))
        for type in types:
            #print("Running {0}".format(type))
            for threads in thread_range:
                #print("Running with {0} threads".format(threads))
                for data_size in data_sizes:
                    for variant in type['variants']:
                        sequential_time = 0
                        variant_time = 0
                        new_time=0

                        for n in range(0,max_n):
                            print("../{0}{1}/{2}.exe".format(platform_paths[platform],config,type['name']));
                            result = subprocess.run(["../{0}{1}/{2}.exe".format(platform_paths[platform],config,type['name']),"-t {0}".format(threads),"-s {0}".format(data_size),"-v {0}".format(variant)],stdout=subprocess.PIPE,universal_newlines=True,cwd="../{0}{1}/".format(platform_paths[platform],config));

                            if result.returncode != 0:
                                print("ERROR {1} returned {0}. Output below.".format(result.returncode, "../{0}{1}/{2}.exe".format(platform_paths[platform],config,type['name'])));
                                print(result.stdout)
                                error_occured = True
                                break;

                            for line in result.stdout.splitlines(): #read and store result in log file
                                line_type = line[0:3]
                                if line_type == "SEQ":
                                    sequential_time += float(line[4:])
                                elif line_type == "VAR":
                                    variant_time += float(line[4:])

                            if variant_time != 0:
                                new_time += sequential_time/variant_time

                            sys.stdout.write("{0: >2} out of {1: >2}\r".format(n+1, max_n))
                            sys.stdout.flush()

                        if error_occured:
                            break;
                        sequential_time = sequential_time/max_n
                        variant_time = variant_time/max_n
                        new_time = new_time/max_n

                        #TODO write file.
                        print("Executed {0: <8} ({1: <11}) size {2: >8} /w {3: >2} thr, seq: {4: >12,.3e}, var: {5: >12,.3e}, new time {6: >12,.3%}".format("{0} {1}".format(type['name'],variant),"{0} {1}".format(platform,config),data_size,threads,sequential_time,variant_time,new_time));
                if error_occured:
                    break;
            if error_occured:
                break;
        if error_occured:
            break;
    if error_occured:
        break;


    #TODO if not error process data and save figures.
if error_occured:
    print("Exited upon error.");
else:
    print("Done.");