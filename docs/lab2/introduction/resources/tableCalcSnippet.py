# Simplified excerpt from Wrapper/GeneratePlots.py
# testName is one of the following 'Contrast' 'Grayscale' 'Histogram' 'Smooth'
# imageName is one of the following 4 9 15
# results is the main result object
# time_data has the averaged timing of each test in nanoseconds, 'cuda' for the CUDA run and seq from the CPU run
# result['prof'] contains all metric from nvprof (from the first profiling run)

data['total-seq-time'] = time_data['seq']['total_time']/1e6; # ms
data['total-cuda-time'] = time_data['cuda']['total_time']/1e6; # ms
data['total-speedup'] = time_data['seq']['total_time']/time_data['cuda']['total_time'];

data['kernel-seq-time'] = time_data['seq']['kernel_time']/1e6; # ms
data['kernel-cuda-time'] = time_data['cuda']['kernel_time']/1e3; # us
data['kernel-speedup'] = time_data['seq']['kernel_time']/time_data['cuda']['kernel_time'];

data['theoretical-gflops'] = MAX_GFLOPS;
data['theoretical-bandwidth'] = MAX_BANDWIDTH;

data['total-mflops'] = (int(result['prof'][testName]['flop_count_sp']['max'])/1e6); # MFLOPS
# Strip of the % sign using [:-1]
data['reported-sp-efficiency'] = float(result['prof'][testName]['flop_sp_efficiency']['max'][:-1])
#This is the old "bad way"
#data['attained-gflops'] = (int(result['prof'][testName]['flop_count_sp']['max'])/1e9)/(time_data['cuda']['kernel_time']/1e9); # GFLOPS/s
# The description for this value (data['reported-sp-efficiency']) is quite literally what we need, it's a percentage
data['attained-gflops'] = MAX_GFLOPS*data['reported-sp-efficiency']/100; # GFLOPS/s
data['attained-bandwidth'] = GetDRAMThroughput(result['prof'][testName],'max'); # GB/s

# Strip of the % sign using [:-1]
data['reported-sm-efficiency'] = float(result['prof'][testName]['sm_efficiency']['max'][:-1])

# GIOPS/s (2e9 because documentation says so, instead of 1e9, something due to issue cycles)
data['attained-iops'] = (int(result['prof'][testName]['inst_integer']['max'][:-1])/2e9)/(time_data['cuda']['kernel_time']/1e9); 