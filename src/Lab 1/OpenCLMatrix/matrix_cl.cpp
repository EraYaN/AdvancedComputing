#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <math.h>
#include <time.h>
#include <omp.h>
#include <direct.h>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <tclap/CmdLine.h>
#include <variant.h>
#include <interactive_tools.h>
#include <opencl_helpers.h>
#include <user_float.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

/****************************************************
the following function calculate the below equation
vector_out = vector_in x matrix_in
***************************************************/
void matrix_mult_sq(int size, user_float_t *matrix1_in,
	user_float_t *matrix2_in, user_float_t *matrix_out) {
	int rowsOut, rowsIn, cols;
	int j;
	for (cols = 0; cols<size; cols++) {
		for (rowsOut = 0; rowsOut<size; rowsOut++) {
			matrix_out[cols + rowsOut*size] = 0.0;
			for (j = 0, rowsIn = 0; rowsIn<size; j++, rowsIn++) {
				matrix_out[cols + rowsOut*size] += matrix1_in[j + rowsOut*size] * matrix2_in[rowsIn*size + cols];
			}
		}
	}
}

/*****************************************************
the following function generates a "size"-element vector
and a "size x size" matrix
****************************************************/
void matrix_gen(unsigned int size, user_float_t *matrix) {
	unsigned int i;
	for (i = 0; i < size*size; i++)
		matrix[i] = ((user_float_t)rand()) / 5307.0;
}

int main(int argc, char *argv[]) {

	// Wrap everything in a try block.  Do this every time,
	// because exceptions will be thrown for problems.
	try {
		// Define the command line object, and insert a message
		// that describes the program. The "Command description message"
		// is printed last in the help text. The second argument is the
		// delimiter (usually space) and the last one is the version number.
		// The CmdLine object parses the argv array based on the Arg objects
		// that it contains.
#ifdef USE_DOUBLES
		TCLAP::CmdLine cmd("OpenCL Matrix x Matrix Multiplication (Double Precision)", ' ', "0.9");
#else
		TCLAP::CmdLine cmd("OpenCL Matrix x Matrix Multiplication (Single Precision)", ' ', "0.9");
#endif

		// Define a value argument and add it to the command line.
		// A value arg defines a flag and a type of value that it expects,
		// such as "-n Bishop".
		TCLAP::ValueArg<unsigned int> threadsArg("t", "threads", "Local workgroup size.", true, 2, "unsigned int");
		TCLAP::ValueArg<unsigned int> datasizeArg("s", "data_size", "Data size.", true, 2, "unsigned int");
		TCLAP::ValueArg<unsigned int> iterationsArg("n", "iterations", "The number of iterations.", false, 1, "unsigned int");
		TCLAP::ValuesConstraint<int> variantConstraint(variants);
		TCLAP::ValueArg<int> variantArg("v", "variant", "Variant ID to run.", false, (int)base, &variantConstraint, false);
		TCLAP::SwitchArg debugArg("d", "debug", "Enable debug mode, verbose output.", false);
		TCLAP::SwitchArg interactiveArg("i", "interactive", "Enable interactive mode.", false);

		// Add the argument nameArg to the CmdLine object. The CmdLine object
		// uses this Arg to parse the command line.
		cmd.add(threadsArg);
		cmd.add(datasizeArg);
		cmd.add(iterationsArg);
		cmd.add(variantArg);
		cmd.add(debugArg);
		cmd.add(interactiveArg);

		// Parse the argv array.
		cmd.parse(argc, argv);

		// Get the value parsed by each arg.
		unsigned int size = datasizeArg.getValue();
		unsigned int localSize = threadsArg.getValue();
		unsigned iterations = iterationsArg.getValue();
		Variant variant = (Variant)variantArg.getValue();
		bool debug = debugArg.getValue();
		bool interactive = interactiveArg.getValue();

		if ((size == 0) || (localSize == 0)) {
			cerr << "Incorrect arguments, make sure both arguments are integers greater than zero." << endl;
			if (interactive) wait_for_input();
			exit(EXIT_OPENCLERROR);
		} else if ((size % localSize) != 0) {
			cerr << "size (" << size << ") should be a multiple of localSize (" << localSize << ")" << endl;
			if (interactive) wait_for_input();
			exit(EXIT_OPENCLERROR);
		}

		if (debug) {
			char* buffer;
			// Get the current working directory:
			if ((buffer = _getcwd(NULL, 0)) == NULL)
				cerr << "_getcwd error" << endl;
			else {
				printf("Current Directory: %s\n", buffer);
				free(buffer);
			}
		}

		user_float_t *matrix1 = (user_float_t *)malloc(sizeof(user_float_t)*size*size);
		user_float_t *matrix2 = (user_float_t *)malloc(sizeof(user_float_t)*size*size);
		user_float_t *result_sq = (user_float_t *)malloc(sizeof(user_float_t)*size*size);
		user_float_t *result_pl = (user_float_t *)malloc(sizeof(user_float_t)*size*size);

		matrix_gen(size, matrix1);
		matrix_gen(size, matrix2);

		double time_sq;
		double time_opencl;


		cl_event mulDone;

		time_sq = omp_get_wtime();
		for (unsigned iteration = 0; iteration < iterations; iteration++) {
			matrix_mult_sq(size, matrix1, matrix2, result_sq);
		}
		time_sq = omp_get_wtime() - time_sq;


		cl_int status;



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

		size_t maxWorkGroupSize;
		// print max work group size
		status |= clGetDeviceInfo(devices[device_id], CL_DEVICE_MAX_WORK_GROUP_SIZE,
			sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);

		if (status != CL_SUCCESS) {
			cerr << "Could not get CL_DEVICE_MAX_WORK_GROUP_SIZE from device: " << getErrorString(status) << endl;
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
			//print max work size.
			printf("Max work group size: %zu\n", maxWorkGroupSize);
		}

		//If the localSize is too much for this device, abort.
		if (localSize*localSize > maxWorkGroupSize) {
			localSize = max(1, (int)floor(sqrt(maxWorkGroupSize)));
			if(debug)
				printf("Given localSize is too big, setting it to maximum of %d\n", localSize);
		}

		cl_ulong globalMemorySize;
		// print global memory size
		clGetDeviceInfo(devices[device_id], CL_DEVICE_GLOBAL_MEM_SIZE,
			sizeof(globalMemorySize), &globalMemorySize, NULL);
		if(debug)
			printf("Global memory size: %lu MiB\n", (unsigned long)round((double)globalMemorySize / 1024 / 1024));

		//If the size is too big for the global memory abort. (3 int sizes for safety and one for the size parameter)
		if (size*size*sizeof(user_float_t)*3 + 4*sizeof(int) > globalMemorySize) {
			if (debug)
				printf("Given data size is too big, the device would run out of memory.\n");
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
		// STEP 5: Create device buffers, images and copy data to buffers
		//-----------------------------------------------------
		cl_mem bufferMatrix1In, bufferMatrix2In, bufferMatrixOut;


		bufferMatrix1In = clCreateBuffer(
			context,
			CL_MEM_READ_ONLY,
			size * size * sizeof(user_float_t),
			NULL,
			&status);

		if (status != CL_SUCCESS) {
			cerr << "error in step 5, creating buffer for bufferA: " << getErrorString(status) << endl;
			if (interactive) wait_for_input();
			exit(EXIT_OPENCLERROR);
		}

		bufferMatrix2In = clCreateBuffer(
			context,
			CL_MEM_READ_ONLY,
			size * size * sizeof(user_float_t),
			NULL,
			&status);

		if (status != CL_SUCCESS) {
			cerr << "error in step 5, creating buffer for bufferB: " << getErrorString(status) << endl;
			if (interactive) wait_for_input();
			exit(EXIT_OPENCLERROR);
		}

		bufferMatrixOut = clCreateBuffer(
			context,
			CL_MEM_WRITE_ONLY,
			size * size * sizeof(user_float_t),
			NULL,
			&status);

		if (status != CL_SUCCESS) {
			cerr << "error in step 5, creating buffer for bufferC: " << getErrorString(status) << endl;
			if (interactive) wait_for_input();
			exit(EXIT_OPENCLERROR);
		}

		char *mulFileName;

#ifdef USE_DOUBLES
		mulFileName = "vectorMatrixMulDouble.cl";
#else
		mulFileName = "vectorMatrixMulFloat.cl";
#endif

		std::fstream kernelFile(mulFileName);
		std::string content(
			(std::istreambuf_iterator<char>(kernelFile)),
			std::istreambuf_iterator<char>()
		);
		size_t mulSize = content.size();
		const char* mulBuffer = new char[mulSize];
		mulBuffer = content.c_str();

		//-----------------------------------------------------
		// STEP 6: Create and compile the program
		//-----------------------------------------------------
		cl_program program = clCreateProgramWithSource(
			context,
			1,
			(const char**)&mulBuffer,
			&mulSize,
			&status);

		//delete mulBuffer;


		// Build (compile) the program for the devices with
		// clBuildProgram()
		const char options[] = "-cl-std=CL1.2";
		status |= clBuildProgram(
			program,
			1,
			&devices[device_id],
			options,
			NULL,
			NULL);

		if (status != CL_SUCCESS) {
			cerr << "error in step 6: " << getErrorString(status) << endl;
			printCLBuildOutput(program, &devices[device_id]);
			if (interactive) wait_for_input();
			exit(EXIT_OPENCLERROR);
		}

		//-----------------------------------------------------
		// STEP 7: Create the kernel
		//-----------------------------------------------------
		cl_kernel mulKernel = NULL;

		// Use clCreateKernel() to create a kernel from the
		mulKernel = clCreateKernel(program, "mul_kernel", &status);
		if (status != CL_SUCCESS) {
			cerr << "error in step 7: " << getErrorString(status) << endl;
			if (interactive) wait_for_input();
			exit(EXIT_OPENCLERROR);
		}


		//TODO figure out if this is fair, all the setup is included now.
		time_opencl = omp_get_wtime();
		for (unsigned iteration = 0; iteration < iterations; iteration++) {
			status = clEnqueueWriteBuffer(
				cmdQueue,
				bufferMatrix1In,
				CL_FALSE,
				0,
				size * size * sizeof(user_float_t),
				matrix1,
				0,
				NULL,
				NULL);

			status |= clEnqueueWriteBuffer(
				cmdQueue,
				bufferMatrix2In,
				CL_FALSE,
				0,
				size * size * sizeof(user_float_t),
				matrix2,
				0,
				NULL,
				NULL);

			if (status != CL_SUCCESS) {
				cerr << "error in step 5, writing data: " << getErrorString(status) << endl;
				if (interactive) wait_for_input();
				exit(EXIT_OPENCLERROR);
			}

			//-----------------------------------------------------
			// STEP 8: Set the kernel arguments
			//-----------------------------------------------------
			// Associate the input and output buffers with the
			// kernel
			// using clSetKernelArg()
			status = clSetKernelArg(
				mulKernel,
				0,
				sizeof(cl_mem),
				&bufferMatrix1In);
			status |= clSetKernelArg(
				mulKernel,
				1,
				sizeof(cl_mem),
				&bufferMatrix2In);
			status |= clSetKernelArg(
				mulKernel,
				2,
				sizeof(cl_mem),
				&bufferMatrixOut);
			status |= clSetKernelArg(
				mulKernel,
				3,
				sizeof(cl_int),
				&size);


			if (status != CL_SUCCESS) {
				cerr << "error in step 8: " << getErrorString(status) << endl;
				if (interactive) wait_for_input();
				exit(EXIT_OPENCLERROR);
			}

			//-----------------------------------------------------
			// STEP 9: Configure the work-item structure
			//-----------------------------------------------------
			// Define an index space (global work size) of work
			// items for
			// execution. A workgroup size (local work size) is not
			// required,
			// but can be used.

			size_t globalWorkSize[2];
			globalWorkSize[0] = size;
			globalWorkSize[1] = size;

			size_t localWorkSize[2];
			localWorkSize[0] = localSize;
			localWorkSize[1] = localSize;

			status |= clEnqueueNDRangeKernel(
				cmdQueue,
				mulKernel,
				2,
				NULL,
				globalWorkSize,
				localWorkSize,
				0,
				NULL,
				&mulDone);

			if (status != CL_SUCCESS) {
				clWaitForEvents(1, &mulDone);
				cerr << "error in clEnqueueNDRangeKernel: " << getErrorString(status) << endl;
				if (interactive) wait_for_input();
				exit(EXIT_OPENCLERROR);
			}

			clEnqueueReadBuffer(
				cmdQueue,
				bufferMatrixOut,
				CL_TRUE,
				0,
				size * size * sizeof(user_float_t),
				result_pl,
				1,
				&mulDone,
				NULL);


			if (status != CL_SUCCESS) {
				cerr << "error in reading data: " << getErrorString(status) << endl;
				exit(EXIT_OPENCLERROR);
			}
		}
		time_opencl = omp_get_wtime() - time_opencl;


		if (debug) {
			printf("ITR:%d\n", iterations);
			printf("DAT:%d\n", size);
			printf("THD:%d\n", localSize);
			//printf("PRC:%d\n", omp_get_num_procs());
		}
		printf("SEQ:%.14f\n", time_sq);
		printf("VAR:%.14f\n", time_opencl);

		if (debug) {
			cout << "matrix1: " << endl;
			printMatrix(matrix1, size, size);
			cout << "matrix2: " << endl;
			printMatrix(matrix2, size, size);
			cout << "result_sq: " << endl;
			printMatrix(result_sq, size, size);
			cout << "result_pl: " << endl;
			printMatrix(result_pl, size, size);

		}

		//check
		bool checkResult = verifyMatrixResult(result_sq, result_pl, size, debug);

		//-----------------------------------------------------
		// STEP 10: Release OpenCL resources
		//-----------------------------------------------------

		// Free OpenCL resources
		clReleaseKernel(mulKernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(cmdQueue);
		clReleaseMemObject(bufferMatrix1In);
		clReleaseMemObject(bufferMatrix2In);
		clReleaseMemObject(bufferMatrixOut);
		clReleaseContext(context);

		//Free up memory and close files
		free(matrix1);
		free(matrix2);
		free(result_sq);
		free(result_pl);

		if (interactive) {
			wait_for_input();
		}

		if (checkResult)
			return EXIT_SUCCESS;
		else
			return EXIT_WRONGVALUE;

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{
		cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
		return EXIT_BADARGUMENT;
	}
}

