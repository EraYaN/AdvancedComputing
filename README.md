# Advanced Computing


# Lab 1

**Prerequisites**

 * Windows 64-bit
 * CUDA Toolkit 8.0
 * Visual Studio 2015 (Community will work)
 * Nsight for Visual Studio 2015
 * Python Tools for Visual Studio
 * Python 3.5.x

**Python packages**

* matplotlib
* numpy
* scipy.optimize
* terminaltables

**Steps**

 1. Open `/src/Lab 1/Lab 1.sln` in VS 2015
 2. Batch build all projects with: Build -> Batch Build... -> Select All -> Rebuild
 3. Run `Wrapper.py` from within Visual Studio for a full data set.
 4. Alternatively, run one program with the --help switch for all support command-line arguments.

# Lab 2

**Prerequisites**

 * Windows 64-bit or Linux 64-bit
 * CUDA Toolkit 7.5 or 8.0 (8.0 preferred)
 * Visual Studio 2015 (Community will work) or a C++11 capable G++
 * Nsight for Visual Studio 2015 (only for Visual Studio)
 * Python Tools for Visual Studio (only for Visual Studio)
 * Python 3.5.x or Python 2.7.x

**Python packages**

* matplotlib
* numpy
* terminaltables
* future
* six
* jinja2

**Steps (Windows)**

 1. Open `/src/Lab 2/Lab 2.sln` in VS 2015
 2. Batch build all projects with: Build -> Batch Build... -> Select All -> Rebuild
 4. Run the benchmark program with the --help switch for all supported command-line arguments.

**Steps (Linux)**

 1. Run `make all` in `/src/Lab 2/`
 2. Run `make schedule` to schedule the benchmark for each the three images
 3. Alternatively, run `make schedule_single RUN_IMAGE="image04.bmp" RUN_ID="image04-custom" PROFILING="no" BENCH_ARGUMENTS="--save-images --shared-histogram-kernel -d"` to schedule the benchmark for `image04.bmp` without profiling and the output images saved to `run_output/image04-custom*.bmp` and other output in `run_output/image04-custom/*`
 4. Alternatively, run `make run` to run the benchmarks for the three images directly and save their images in `run_output/`
 5. Alternatively, run `out/benchmark --help` to get all other options.
 6. Alternatively, run `python Wrapper/Wrapper.py --disable-plot` to get the full data set.
 6. Lastly run `python Wrapper/Wrapper.py --disable-bench` in the folder with the output `*.pickle` files to generate all the figures.

**Note:** To use the `rdtsc()` intrinsic for timing instead of the default `chrono::high_resolution_clock` append `USE_RDTSC=1` to any of the make command, results may vary. See `/src/acs-shared/timing.*` for implementation details.
**Note:** `make runserver` is an alias for `make schedule`
**Note:** There is a helper rsync script in the `/scripts/` folder

# Lab 3

**Prerequisites**

 * Windows 64-bit or Linux 64-bit
 * CUDA Toolkit 7.5 or 8.0 (8.0 preferred)
 * Visual Studio 2015 (Community will work) or a C++11 capable G++
 * Nsight for Visual Studio 2015 (only for Visual Studio)
 * Python Tools for Visual Studio (only for Visual Studio)
 * Python 3.5.x or Python 2.7.x

**Python packages**

* matplotlib
* numpy
* terminaltables
* future
* six
* jinja2

**Steps (Windows)**

 1. Open `/src/Lab 3/Lab 3.sln` in VS 2015
 2. Batch build all projects with: Build -> Batch Build... -> Select All -> Rebuild
 4. Run the benchmark programs directly on the command-line.

**Steps (Linux)**

 1. Run `make all` in `/src/Lab 3/`
 2. Run `make schedule` to schedule all the benchmarks.
 3. Alternatively, run `make schedule_cpu RUN_ID="cpu-64x64-custom" PROFILING="no" BENCH_ARGUMENTS="" NETWORK_SIZE=64` to schedule the CPU benchmark for a brain size of 32 x 32 cells without profiling.
 4. Alternatively, run `make run` to run the benchmarks directly.
 6. Alternatively, run `python Wrapper/Wrapper.py --disable-plot` to get the full data set.
 6. Lastly run `python Wrapper/Wrapper.py --disable-bench` in the folder with the output `*.pickle` files to generate all the figures.

**Note:** `make runserver` is an alias for `make schedule`
**Note:** There is a helper rsync script in the `/scripts/` folder