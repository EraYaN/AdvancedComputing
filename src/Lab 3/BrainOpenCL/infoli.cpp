/*
 *
 * Copyright (c) 2012, Neurasmus B.V., The Netherlands,
 * web: www.neurasmus.com email: info@neurasmus.com
 *
 * Any use or reproduction in whole or in parts is prohibited
 * without the written consent of the copyright owner.
 *
 * All Rights Reserved.
 *
 *
 * Author: Sebastian Isaza
 * Created: 19-01-2012
 * Modified: 07-08-2012
 *
 * Description: Top source file of the Inferior Olive model, originally written
 * in Matlab by Jornt De Gruijl. It contains the implementation of all functions.
 * The main function allocates the necessary memory, initializes the system
 * state and runs the model calculations.
 *
 */
#include "infoli.h"
#include "init.h"

#define CSV_SEPARATOR ','
#define LINE_MARKER '@'

using namespace std;

int main(int argc, char *argv[]){
#ifdef _WIN32
	_putenv_s("CUDA_CACHE_DISABLE", "1");
#else
	setenv("CUDA_CACHE_DISABLE", "1", 1);
#endif
    const char *outFileName = "../InferiorOlive_Output_OpenCL.txt";
	cl_uint i, j, k, p, q;
	i = 0;
    int simSteps = 0;
    int simTime = 0;
    int inputFromFile = 0;
    int initSteps;
    user_float_t *cellStatePtr;
	user_float_t *cellVDendPtr;
	user_float_t iApp = 0;

	bool debug = true;
	bool interactive = true;

    int seedvar;
    char temp[100];//warning: this buffer may overflow
    perftime_t t0, t1, tNeighbourStart, tNeighbourEnd, tComputeStart, tComputeEnd, tReadStart, tReadEnd, tWriteStateStart, tWriteStateEnd, tWriteCompStart, tWriteCompEnd, tInitStart, tInitEnd, tLoopStart, tLoopEnd, tWriteFileStart, tWriteFileEnd;
	double tNeighbour, tCompute, tUpdate, tRead, tWriteFile, tInit, tLoop, usecs;
    tNeighbour = tCompute = tUpdate = tRead = tWriteFile = tInit = tLoop = 0;

    cl_event writeDone, neighbourDone, computeDone, readDone;
    cl_int status = 0;
	cl_int statusNeighbour = 0;
	cl_int statusCompute = 0;

    t0 = now();
    if(EXTRA_TIMING){
        tInitStart = now();
    }

    simTime = SIMTIME; // in miliseconds
    simSteps = ceil(simTime/DELTA);

    DEBUG_PRINT(("Inferior Olive Model (%d x %d cell mesh)\n", IO_NETWORK_DIM1, IO_NETWORK_DIM2));

	ofstream pOutFile;

	if (WRITE_OUTPUT) {
		//Open output file
		pOutFile.open(outFileName);

		if (!pOutFile) {
			cerr << "Error: Couldn't create " << outFileName << endl;
			exit(EXIT_FAILURE);
		}
		pOutFile << "#simSteps Time(ms) Input(Iapp) Output(V_axon)" << endl;
	}

    //Malloc for the array of cellStates and cellCompParams
    mallocCells(&cellVDendPtr, &cellStatePtr);

    //Write initial state values
    InitState(cellStatePtr, cellVDendPtr);

	//Initialize g_CaL
    init_g_CaL(cellStatePtr);

	//-----------------------------------------------------
	// STEP 1: Discover and initialize the platforms
	//-----------------------------------------------------
	cl_uint numPlatforms = 0;
	cl_platform_id *platforms = NULL;

	// Use clGetPlatformIDs() to retrieve the number of
	// platforms
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	// Allocate enough space for each platform
	platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));

	// Fill in platforms with clGetPlatformIDs()
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);

	if (status != CL_SUCCESS) {
		cerr << "error in step 1: " << getErrorString(status) << endl;
		if (interactive) wait_for_input();
		exit(EXIT_OPENCLERROR);
	}

	//-----------------------------------------------------
	// STEP 2: Discover and initialize the devices
	//-----------------------------------------------------
	cl_uint numDevices = 0;
	cl_device_id *devices = NULL;

	// Use clGetDeviceIDs() to retrieve the number of
	// devices present
	status = clGetDeviceIDs(
		platforms[0],
		CL_DEVICE_TYPE_ALL,
		0,
		NULL,
		&numDevices);
	// Allocate enough space for each device
	devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));

	// Fill in devices with clGetDeviceIDs()
	status |= clGetDeviceIDs(
		platforms[0],
		CL_DEVICE_TYPE_ALL,
		numDevices,
		devices,
		NULL);
	// select the device which will be used
	int device_id = 0;

	if ((status != CL_SUCCESS) || ((unsigned int)device_id >= numDevices)) {
		cerr << "error in step 2: " << getErrorString(status) << endl;
		if (interactive) wait_for_input();
		exit(EXIT_OPENCLERROR);
	}


	if (debug) {
		// print device name
		char* value;
		size_t valueSize;
		clGetDeviceInfo(devices[device_id], CL_DEVICE_NAME, 0, NULL, &valueSize);
		value = (char*)malloc(valueSize);
		clGetDeviceInfo(devices[device_id], CL_DEVICE_NAME, valueSize, value, NULL);
		printf("Running on device: %s\n", value);
		free(value);
	}

	//-----------------------------------------------------
	// STEP 3: Create a context
	//-----------------------------------------------------
	cl_context context = NULL;

	// Create a context using clCreateContext() and
	// associate it with the devices
	context = clCreateContext(
		NULL,
		1,
		&devices[device_id],
		NULL,
		NULL,
		&status);

	if (status != CL_SUCCESS) {
		cerr << "error in step 3: " << getErrorString(status) << endl;
		if (interactive) wait_for_input();
		exit(EXIT_OPENCLERROR);
	}


	//-----------------------------------------------------
	// STEP 4: Create a command queue
	//-----------------------------------------------------
	cl_command_queue cmdQueue;

	// Create a command queue using clCreateCommandQueue(),
	// and associate it with the device you want to execute
	// on
	cmdQueue = clCreateCommandQueue(
		context,
		devices[device_id],
		0,
		&status);

	if (status != CL_SUCCESS) {
		cerr << "error in step 4: " << endl;
		if (interactive) wait_for_input();
		exit(EXIT_OPENCLERROR);
	}


    //-----------------------------------------------------
    // STEP 5: Create device buffers
    //-----------------------------------------------------
	cl_mem buffer_cellStatePtr;
	cl_mem buffer_cellVDendPtr;
	cl_mem buffer_iApp;

	buffer_cellStatePtr = clCreateBuffer(
		context,
		CL_MEM_READ_WRITE,
		IO_NETWORK_SIZE*PARAM_SIZE * sizeof(user_float_t),
		NULL,
		&status);

	if (status != CL_SUCCESS) {
		cerr << "error in step 5, creating buffer for cellStatePtr: " << getErrorString(status) << endl;
		if (interactive) wait_for_input();
		exit(EXIT_OPENCLERROR);
	}

	buffer_iApp = clCreateBuffer(
		context,
		CL_MEM_READ_WRITE,
		simSteps * sizeof(user_float_t),
		NULL,
		&status);

	if (status != CL_SUCCESS) {
		cerr << "error in step 5, creating buffer for iApp: " << getErrorString(status) << endl;
		if (interactive) wait_for_input();
		exit(EXIT_OPENCLERROR);
	}

	buffer_cellVDendPtr = clCreateBuffer(
		context,
		CL_MEM_READ_WRITE,
		IO_NETWORK_SIZE * sizeof(user_float_t),
		NULL,
		&status);

	if (status != CL_SUCCESS) {
		cerr << "error in step 5, creating image for cellVDendPtr: " << getErrorString(status) << endl;
		if (interactive) wait_for_input();
		exit(EXIT_OPENCLERROR);
	}

    //-----------------------------------------------------
    // STEP 6: Write host data to device buffers
    //-----------------------------------------------------
	/**
	** Please check if your kernel files can be opened,
	** especially when running on the server
	**/
	const char *computeFileName = "compute_kernel.cl";

	// computeFile
	ifstream computeFile(computeFileName);
	if (!computeFile) {
		cerr << "cannot open compute file" << endl;
		cerr << "current path:" << computeFileName << endl;
		if (interactive) wait_for_input();
		exit(EXIT_FAILURE);
	}

	std::string contentCompute(
		(std::istreambuf_iterator<char>(computeFile)),
		std::istreambuf_iterator<char>()
	);
	size_t computeSize = contentCompute.size();
	const char* computeBuffer = new char[computeSize];
	computeBuffer = contentCompute.c_str();


    //-----------------------------------------------------
    // STEP 7: Create and compile the program
    //-----------------------------------------------------

	cl_program programCompute = clCreateProgramWithSource(
		context,
		1,
		(const char**)&computeBuffer,
		&computeSize,
		&statusCompute);

	//delete mulBuffer;

	// Build (compile) the program for the devices with
	// clBuildProgram()
	const char options[] = "-cl-std=CL1.2 -cl-finite-math-only -cl-denorms-are-zero";

	statusCompute |= clBuildProgram(
		programCompute,
		1,
		&devices[device_id],
		options,
		NULL,
		NULL);

	if (statusCompute != CL_SUCCESS) {
		cerr << "error in step 7: " << getErrorString(statusCompute) << endl << "Build output below:" << endl;
		printCLBuildOutput(programCompute, &devices[device_id]);
		cerr << "End of build log." << endl;
		if (interactive) wait_for_input();
		exit(EXIT_OPENCLERROR);
	}

    //-----------------------------------------------------
    // STEP 8: Create the kernel
    //----------------------------------------------------
	cl_kernel computeKernel = NULL;

	// Use clCreateKernel() to create a kernel from the
	computeKernel = clCreateKernel(programCompute, "compute_kernel", &statusCompute);
	if (statusCompute != CL_SUCCESS) {
		cerr << "error in step 8: " << getErrorString(statusCompute) << endl;
		if (interactive) wait_for_input();
		exit(EXIT_OPENCLERROR);
	}

	status = clEnqueueWriteBuffer(
		cmdQueue,
		buffer_cellStatePtr,
		CL_FALSE,
		0,
		IO_NETWORK_SIZE*PARAM_SIZE * sizeof(user_float_t),
		cellStatePtr,
		0,
		NULL,
		NULL);

	status = clEnqueueWriteBuffer(
		cmdQueue,
		buffer_cellVDendPtr,
		CL_FALSE,
		0,
		IO_NETWORK_SIZE * sizeof(user_float_t),
		cellVDendPtr,
		0,
		NULL,
		NULL);

	if (status != CL_SUCCESS) {
		cerr << "error in step 8, writing data: " << getErrorString(status) << endl;
		if (interactive) wait_for_input();
		exit(EXIT_OPENCLERROR);
	}


    //-----------------------------------------------------
    // STEP 9: Set the kernel arguments
    //-----------------------------------------------------
	// Associate the input and output buffers with the
	// kernel
	// using clSetKernelArg()
	// compute arguments
	statusCompute = clSetKernelArg(
		computeKernel,
		0,
		sizeof(cl_mem),
		&buffer_cellStatePtr);

	if (statusCompute != CL_SUCCESS) {
		cerr << "error in step 9.1a: " << getErrorString(statusCompute) << endl;
		if (interactive) wait_for_input();
		exit(EXIT_OPENCLERROR);
	}

	statusCompute = clSetKernelArg(
		computeKernel,
		1,
		sizeof(cl_mem),
		(void*)&buffer_cellVDendPtr);


	if (statusCompute != CL_SUCCESS) {
		cerr << "error in step 9.1b: " << getErrorString(statusCompute) << endl;
		if (interactive) wait_for_input();
		exit(EXIT_OPENCLERROR);
	}



	if (EXTRA_TIMING) {
		tInitEnd = tLoopStart = now();
	}

    //-----------------------------------------------------
    // STEP 10: Configure the work-item structure
    //-----------------------------------------------------

	statusCompute = clSetKernelArg(
		computeKernel,
		2,
		sizeof(user_float_t),
		&iApp);

	if (statusCompute != CL_SUCCESS) {
		cerr << "error before loop, compute clSetKernelArg. Error: " << getErrorString(statusCompute) << endl;
		if (interactive) wait_for_input();
		exit(EXIT_FAILURE);
	}

    for(i=0;i<simSteps;i++) {
		if (WRITE_OUTPUT) {
			pOutFile << string_format("%d %.2f %.1f ", i + 1, i*0.05, iApp);
		}


        //-----------------------------------------------------
        // STEP 11.1: Run neighbour kernel
        //-----------------------------------------------------

		size_t globalWorkSize[2];
		globalWorkSize[0] = IO_NETWORK_DIM1;
		globalWorkSize[1] = IO_NETWORK_DIM2;// size;

		size_t localWorkSize[2];
		localWorkSize[0] = min(16, IO_NETWORK_DIM1);// localSize;
		localWorkSize[1] = min(16, IO_NETWORK_DIM2);// localSize;


        if(EXTRA_TIMING){
            tComputeStart = now();
        }

        //-----------------------------------------------------
        // STEP 11.2: Run compute kernel
        //-----------------------------------------------------

		statusCompute = clEnqueueNDRangeKernel(
			cmdQueue,
			computeKernel,
			2,
			NULL,
			globalWorkSize,
			localWorkSize,
			0,
			NULL,
			&computeDone);

		if (statusCompute != CL_SUCCESS) {
			cerr << "error in loop, compute clEnqueueNDRangeKernel. Error: " << getErrorString(statusCompute) << endl;
			if (interactive) wait_for_input();
			exit(EXIT_FAILURE);
		}

        if(EXTRA_TIMING){
            statusCompute = clWaitForEvents(1, &computeDone);
            tComputeEnd = now();
            tCompute += diffToNanoseconds(tComputeStart, tComputeEnd);
            tReadStart = now();

			if (statusCompute != CL_SUCCESS) {
				cerr << "error in loop, compute clWaitForEvents. Error: " << getErrorString(statusCompute) << endl;
				if (interactive) wait_for_input();
				exit(EXIT_FAILURE);
			}
        }

        if(statusCompute != CL_SUCCESS){
            cerr << "error in loop, compute. Error: " << getErrorString(statusCompute) << endl;
			if (interactive) wait_for_input();
            exit(EXIT_FAILURE);
        }


        // transfer data from device to CPU
        if(WRITE_OUTPUT){
            //-----------------------------------------------------
            // STEP 11.3: Read output data from device
            //-----------------------------------------------------
			clEnqueueReadBuffer(
				cmdQueue,
				buffer_cellStatePtr,
				CL_TRUE,
				0,
				IO_NETWORK_SIZE*PARAM_SIZE * sizeof(user_float_t),
				cellStatePtr,
				1,
				&computeDone,
				NULL);
        }

        if(EXTRA_TIMING){
            tReadEnd = now();
            tRead += diffToNanoseconds(tReadStart, tReadEnd);
        }
        // write output to file
        if(WRITE_OUTPUT){
            if(EXTRA_TIMING){
                tWriteFileStart = now();
            }
			for (int b = 0; b < IO_NETWORK_SIZE; b++) {
				pOutFile << setprecision(8) << cellStatePtr[b*PARAM_SIZE + STATEADD + AXON_V] << " ";
			}
			pOutFile << endl;
            if(EXTRA_TIMING){
                tWriteFileEnd = now();
                tWriteFile += diffToNanoseconds(tWriteFileStart, tWriteFileEnd);
            }
        }

		//wait_for_input();
		//if (i >= 2) break;
    }
    if(EXTRA_TIMING){
        tLoopEnd = now();
    }

    t1 = now();
    usecs = diffToNanoseconds(t0,t1)/1e9;// / 1000000;

	if (EXTRA_TIMING) {
		tInit = diffToNanoseconds(tInitStart, tInitEnd);
		tLoop = diffToNanoseconds(tLoopStart, tLoopEnd);

		DEBUG_PRINT(("\n"));
		DEBUG_PRINT(("----------------------------------\n"));
		DEBUG_PRINT(("tInit: \t\t %.1f s\n", tInit / 1e9));
		DEBUG_PRINT(("tLoop: \t\t %.1f s\n", tLoop / 1e9));
		DEBUG_PRINT(("\ttNeighbour: \t %.1f s\n", tNeighbour / 1e9));
		DEBUG_PRINT(("\ttCompute: \t %.1f s\n", tCompute / 1e9));
		DEBUG_PRINT(("\ttRead: \t\t %.1f s\n", tRead / 1e9));
		DEBUG_PRINT(("\ttWriteFile: \t %.1f s\n", tWriteFile / 1e9));
		DEBUG_PRINT(("\t----------- + \n"));
		DEBUG_PRINT(("\ttSumLoop: \t %.1f s\n", (tWriteFile + tCompute + tNeighbour + tRead) / 1e9));
		DEBUG_PRINT(("----------------------------------\n"));
		DEBUG_PRINT(("tSum: \t %.1f s\n", (tInit + tLoop) / 1e9));

		cout << LINE_MARKER << "OpenCL" << CSV_SEPARATOR << tInit / 1e9 << CSV_SEPARATOR << tNeighbour / 1e9 << CSV_SEPARATOR << tCompute / 1e9 << CSV_SEPARATOR << (tWriteFile + tRead) / 1e9 << CSV_SEPARATOR << (tInit + tLoop) / 1e9 << endl;
	}

    DEBUG_PRINT(("%d ms of brain time in %d simulation steps\n", simTime, simSteps));
    DEBUG_PRINT((" %.1f s real time \n", usecs));



    //-----------------------------------------------------
    // STEP 12: Release OpenCL resources
    //-----------------------------------------------------
	//clReleaseKernel(neighbourKernel);
	clReleaseKernel(computeKernel);
	//clReleaseProgram(programNeighbour);
	clReleaseProgram(programCompute);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(buffer_cellStatePtr);
	clReleaseMemObject(buffer_cellVDendPtr);
	clReleaseContext(context);

    //Free up memory and close files
    free(cellStatePtr);
    free(cellVDendPtr);
	if (WRITE_OUTPUT) {
		pOutFile.close();
	}

	if(interactive) wait_for_input();

    return EXIT_SUCCESS;
}

