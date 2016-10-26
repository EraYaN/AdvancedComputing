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
	user_float_t iApp;

	bool debug = true;
	bool interactive = false;

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

    //Open output file
	ofstream pOutFile(outFileName);

	if (!pOutFile) {
		cerr << "Error: Couldn't create " << outFileName << endl;
		exit(EXIT_FAILURE);
	}
	pOutFile << "#simSteps Time(ms) Input(Iapp) Output(V_axon)" << endl;

    //Malloc for the array of cellStates and cellCompParams
    mallocCells(&cellVDendPtr, &cellStatePtr);

    //Write initial state values
    InitState(cellStatePtr, cellVDendPtr);

	//Initialize g_CaL
    init_g_CaL(cellStatePtr);

	//for (int ind = 0; ind < IO_NETWORK_SIZE; ind++) {
	//	printf("(state-cpu) dendv %d: %lf\n", ind, cellStatePtr[ind*PARAM_SIZE+ STATEADD + DEND_V]);
	//	printf("(state-cpu) dend_ca2 %d: %lf\n", ind, cellStatePtr[ind*PARAM_SIZE + STATEADD + DEND_CA2]);
	//}

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
	cl_mem buffer_cellStatePtr, buffer_cellVDendPtr;

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

	/*cl_image_format img_format;
	img_format.image_channel_order = CL_INTENSITY;
	img_format.image_channel_data_type = CL_FLOAT;
	buffer_cellVDendPtr = clCreateImage2D(
		context,

		);*/

	buffer_cellVDendPtr = clCreateBuffer(
		context,
		CL_MEM_READ_WRITE,
		IO_NETWORK_SIZE * sizeof(user_float_t),
		NULL,
		&status);

	if (status != CL_SUCCESS) {
		cerr << "error in step 5, creating buffer for cellStatePtr: " << getErrorString(status) << endl;
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
	char *computeFileName, *neighbourFileName;
	neighbourFileName = "neighbour_kernel.cl";
	computeFileName = "compute_kernel.cl";

	// neighbourFile
	ifstream neighbourFile(neighbourFileName);
	if (!neighbourFile) {
		cerr << "cannot open neighbour file" << endl;
		cerr << "current path:" << neighbourFileName << endl;
		exit(EXIT_FAILURE);
	}

	std::string contentNeighbour(
		(std::istreambuf_iterator<char>(neighbourFile)),
		std::istreambuf_iterator<char>()
	);
	size_t neighbourSize = contentNeighbour.size();
	const char* neighbourBuffer = new char[neighbourSize];
	neighbourBuffer = contentNeighbour.c_str();

	// computeFile
	ifstream computeFile(computeFileName);
	if (!computeFile) {
		cerr << "cannot open neighbour file" << endl;
		cerr << "current path:" << neighbourFileName << endl;
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
	cl_program programNeighbour = clCreateProgramWithSource(
		context,
		1,
		(const char**)&neighbourBuffer,
		&neighbourSize,
		&statusNeighbour);

	cl_program programCompute = clCreateProgramWithSource(
		context,
		1,
		(const char**)&computeBuffer,
		&computeSize,
		&statusCompute);

	//delete mulBuffer;

	// Build (compile) the program for the devices with
	// clBuildProgram()
	const char options[] = "-cl-std=CL1.2";
	statusNeighbour |= clBuildProgram(
		programNeighbour,
		1,
		&devices[device_id],
		options,
		NULL,
		NULL);

	if (statusNeighbour != CL_SUCCESS) {
		cerr << "error in step 6: " << getErrorString(statusNeighbour) << endl;
		printCLBuildOutput(programNeighbour, &devices[device_id]);
		if (interactive) wait_for_input();
		exit(EXIT_OPENCLERROR);
	}

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
	cl_kernel neighbourKernel = NULL;

	// Use clCreateKernel() to create a kernel from the
	neighbourKernel = clCreateKernel(programNeighbour, "neighbour_kernel", &statusNeighbour);
	if (statusNeighbour != CL_SUCCESS) {
		cerr << "error in step 8: " << getErrorString(statusNeighbour) << endl;
		if (interactive) wait_for_input();
		exit(EXIT_OPENCLERROR);
	}

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

	status |= clEnqueueWriteBuffer(
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
	statusCompute |= clSetKernelArg(
		computeKernel,
		1,
		sizeof(cl_mem),
		&buffer_cellVDendPtr);



	if (statusCompute != CL_SUCCESS) {
		cerr << "error in step 9.1: " << getErrorString(statusCompute) << endl;
		if (interactive) wait_for_input();
		exit(EXIT_OPENCLERROR);
	}

	// neighbour argument
	statusNeighbour = clSetKernelArg(
		neighbourKernel,
		0,
		sizeof(cl_mem),
		&buffer_cellStatePtr);

	statusNeighbour |= clSetKernelArg(
		neighbourKernel,
		1,
		sizeof(cl_mem),
		&buffer_cellVDendPtr);

	if (statusNeighbour != CL_SUCCESS) {
		cerr << "error in step 9.2 " << getErrorString(statusNeighbour) << endl;
		if (interactive) wait_for_input();
		exit(EXIT_OPENCLERROR);
	}

	if (EXTRA_TIMING) {
		tInitEnd = tLoopStart = now();
	}

    //-----------------------------------------------------
    // STEP 10: Configure the work-item structure
    //-----------------------------------------------------


    for(i=0;i<simSteps;i++) {
        //Compute one sim step for all cells
        if(i>20000-1 && i<20500-1){ iApp = 6;} // start @ 1 because skipping initial values
        else{ iApp = 0;}
		pOutFile << string_format("%d %.2f %.1f ", i + 1, i*0.05, iApp);

        if(EXTRA_TIMING){
            tNeighbourStart = now();
        }

        //-----------------------------------------------------
        // STEP 11.1: Run neighbour kernel
        //-----------------------------------------------------
		// FIX: sizes
		size_t globalWorkSize[2];
		globalWorkSize[0] = IO_NETWORK_DIM1;
		globalWorkSize[1] = IO_NETWORK_DIM2;// size;

		size_t localWorkSize[2];
		localWorkSize[0] = 16;// localSize;
		localWorkSize[1] = 16;// localSize;


		statusNeighbour |= clEnqueueNDRangeKernel(
			cmdQueue,
			neighbourKernel,
			2,
			NULL,
			globalWorkSize,
			localWorkSize,
			0,
			NULL,
			&neighbourDone);


        if(EXTRA_TIMING){
            statusNeighbour = clWaitForEvents(1, &neighbourDone);
            tNeighbourEnd = now();
            tNeighbour += diffToNanoseconds(tNeighbourStart, tNeighbourEnd);
            tComputeStart = now();
        }

		if (statusNeighbour != CL_SUCCESS) {
			cerr << "error in loop, neighbour. Error: " << getErrorString(statusNeighbour) << endl;
			exit(EXIT_FAILURE);
		}

        //-----------------------------------------------------
        // STEP 11.2: Run compute kernel
        //-----------------------------------------------------

		statusCompute = clSetKernelArg(
			computeKernel,
			2,
			sizeof(user_float_t),
			&iApp);

		if (statusCompute != CL_SUCCESS) {
			cerr << "error in loop, compute clSetKernelArg. Error: " << getErrorString(statusCompute) << endl;
			exit(EXIT_FAILURE);
		}

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
			exit(EXIT_FAILURE);
		}

        if(EXTRA_TIMING){
            statusCompute = clWaitForEvents(1, &computeDone);
            tComputeEnd = now();
            tCompute += diffToNanoseconds(tComputeStart, tComputeEnd);
            tReadStart = now();

			if (statusCompute != CL_SUCCESS) {
				cerr << "error in loop, compute clWaitForEvents. Error: " << getErrorString(statusCompute) << endl;
				exit(EXIT_FAILURE);
			}
        }

        if(statusCompute != CL_SUCCESS){
            cerr << "error in loop, compute. Error: " << getErrorString(statusCompute) << endl;
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
            /*for(j = 0; j < IO_NETWORK_DIM1; j++){
                for(k = 0; k < IO_NETWORK_DIM2; k++){
					pOutFile << setprecision(8) << cellStatePtr[(((j + k*IO_NETWORK_DIM1))* PARAM_SIZE) + AXON_V] << " ";
                }
            }*/
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
    usecs = diffToNanoseconds(t0,t1)/1e6;// / 1000000;
    DEBUG_PRINT(("%d ms of brain time in %d simulation steps\n", simTime, simSteps));
    DEBUG_PRINT((" %.1f msecs real time \n", usecs));

    if(EXTRA_TIMING){
        tInit = diffToNanoseconds(tInitStart, tInitEnd);
        tLoop = diffToNanoseconds(tLoopStart, tLoopEnd);

        DEBUG_PRINT(("\n"));
        DEBUG_PRINT(("----------------------------------\n"));
        DEBUG_PRINT(("tInit: \t\t %.1f s\n", tInit/1e9));
        DEBUG_PRINT(("tLoop: \t\t %.1f s\n", tLoop / 1e9));
        DEBUG_PRINT(("\ttNeighbour: \t %.1f s\n", tNeighbour / 1e9));
        DEBUG_PRINT(("\ttCompute: \t %.1f s\n", tCompute / 1e9));
        DEBUG_PRINT(("\ttRead: \t\t %.1f s\n", tRead / 1e9));
        DEBUG_PRINT(("\ttWriteFile: \t %.1f s\n", tWriteFile / 1e9));
        DEBUG_PRINT(("\t----------- + \n"));
        DEBUG_PRINT(("\ttSumLoop: \t %.1f s\n", (tWriteFile + tCompute + tNeighbour + tRead) / 1e9));
        DEBUG_PRINT(("----------------------------------\n"));
        DEBUG_PRINT(("tSum: \t %.1f s\n", (tInit + tLoop) / 1e9));
    }

    //-----------------------------------------------------
    // STEP 12: Release OpenCL resources
    //-----------------------------------------------------
	clReleaseKernel(neighbourKernel);
	clReleaseKernel(computeKernel);
	clReleaseProgram(programNeighbour);
	clReleaseProgram(programCompute);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(buffer_cellStatePtr);
	clReleaseMemObject(buffer_cellVDendPtr);
	clReleaseContext(context);

    //Free up memory and close files
    free(cellStatePtr);
    free(cellVDendPtr);
    pOutFile.close();

	if(interactive) wait_for_input();

    return EXIT_SUCCESS;
}

