import subprocess
import sys

# Program Definitions

max_n = 50 # Times to run program to get average

platforms = ['x86','x64'] # Platform names
platform_paths = {'x86':'','x64':'x64/'} # Platform to Directory Dictionary
configs = ['Release','Debug'] # Build config names
types = ['OpenMP','SSE'] # Program names
max_thread = 10
thread_range = [2,4,8] #range(1,max_thread+1)
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
                    sequential_time = 0
                    parallel_time = 0
                    new_time=0
                    for n in range(0,max_n):
                        #print("Running with data size {0}".format(data_size))
                        result = subprocess.run(["../{0}{1}/{2}.exe".format(platform_paths[platform],config,type),"{0}".format(threads),"{0}".format(data_size)],stdout=subprocess.PIPE,universal_newlines=True,cwd="../{0}{1}/".format(platform_paths[platform],config));   
                        
                        if result.returncode != 0:
                            print("ERROR {1} returned {0}. Output below.".format(result.returncode, "../{0}{1}/{2}.exe".format(platform_paths[platform],config,type)));
                            print(result.stdout)
                            error_occured = True
                            break;

                        for line in result.stdout.splitlines(): #read and store result in log file
                            line_type = line[0:3]                    
                            if line_type == "SEQ":
                                sequential_time += float(line[4:])
                            elif line_type == "PAR":
                                parallel_time += float(line[4:])
                                                   
                        if parallel_time != 0:
                            new_time += sequential_time/parallel_time

                        sys.stdout.write("{0: >2} out of {1: >2}\r".format(n+1, max_n))
                        sys.stdout.flush()

                    if error_occured:
                        break;    
                    sequential_time = sequential_time/max_n
                    parallel_time = parallel_time/max_n
                    new_time = new_time/max_n

                    #TODO write file.
                    print("Executed {0: <6} ({1: <11}) size {2: >8} /w {3: >2} thr, seq: {4: >12,.3e}, par: {5: >12,.3e}, new time {6: >12,.3%}".format(type,"{0} {1}".format(platform,config),data_size,threads,sequential_time,parallel_time,new_time));
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