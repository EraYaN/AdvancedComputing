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
#include <sequential_functions.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

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
		TCLAP::CmdLine cmd("OpenCL Matrix x Vector Multiplication (Double Precision)", ' ', "0.9");
#else
		TCLAP::CmdLine cmd("OpenCL Matrix x Vector Multiplication (Single Precision)", ' ', "0.9");
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
				printf("%s \nLength: %zu\n", buffer, strlen(buffer));
				free(buffer);
			}
		}

		user_float_t *vector = (user_float_t *)malloc(sizeof(user_float_t)*size);
		user_float_t *matrix = (user_float_t *)malloc(sizeof(user_float_t)*size*size);
		user_float_t *result_sq = (user_float_t *)malloc(sizeof(user_float_t)*size);
		user_float_t *result_pl = (user_float_t *)malloc(sizeof(user_float_t)*size);
		matrix_gen(size, matrix);
		vector_gen(size, vector);

		double time_sq;
		double time_opencl;


		cl_event mulDone;

		time_sq = omp_get_wtime();
		for (unsigned iteration = 0; iteration < iterations; iteration++) {
			mv_mult_sq(size, vector, matrix, result_sq);
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
		// STEP 5: Create device buffers, images and copy data to buffers
		//-----------------------------------------------------
		cl_mem bufferVectorIn, bufferMatrixIn, bufferVectorOut;


		bufferVectorIn = clCreateBuffer(
			context,
			CL_MEM_READ_ONLY,
			size * sizeof(user_float_t),
			NULL,
			&status);

		if (status != CL_SUCCESS) {
			cerr << "error in step 5, creating buffer for bufferA: " << getErrorString(status) << endl;
			if (interactive) wait_for_input();
			exit(EXIT_OPENCLERROR);
		}

		bufferMatrixIn = clCreateBuffer(
			context,
			CL_MEM_READ_ONLY,
			size*size * sizeof(user_float_t),
			NULL,
			&status);

		if (status != CL_SUCCESS) {
			cerr << "error in step 5, creating buffer for bufferB: " << getErrorString(status) << endl;
			if (interactive) wait_for_input();
			exit(EXIT_OPENCLERROR);
		}

		bufferVectorOut = clCreateBuffer(
			context,
			CL_MEM_WRITE_ONLY,
			size*size * sizeof(user_float_t),
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
				bufferVectorIn,
				CL_FALSE,
				0,
				size * sizeof(user_float_t),
				vector,
				0,
				NULL,
				NULL);

			status |= clEnqueueWriteBuffer(
				cmdQueue,
				bufferMatrixIn,
				CL_FALSE,
				0,
				size*size * sizeof(user_float_t),
				matrix,
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
				&bufferVectorIn);
			status |= clSetKernelArg(
				mulKernel,
				1,
				sizeof(cl_mem),
				&bufferMatrixIn);
			status |= clSetKernelArg(
				mulKernel,
				2,
				sizeof(cl_mem),
				&bufferVectorOut);
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

			size_t globalWorkSize[1];
			globalWorkSize[0] = size;

			size_t localWorkSize[1];
			localWorkSize[0] = localSize;


			status |= clEnqueueNDRangeKernel(
				cmdQueue,
				mulKernel,
				1,
				NULL,
				globalWorkSize,
				localWorkSize,
				0,
				NULL,
				&mulDone);

			if (status != CL_SUCCESS) {
				clWaitForEvents(1, &mulDone);
				cerr << "error in clEnqueueNDRangeKernel" << endl;
				if (interactive) wait_for_input();
				exit(EXIT_OPENCLERROR);
			}

			clEnqueueReadBuffer(
				cmdQueue,
				bufferVectorOut,
				CL_TRUE,
				0,
				size * sizeof(user_float_t),
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
			cout << "vector: " << endl;
			printArray(vector, size);
			cout << "matrix: " << endl;
			printMatrix(matrix, size, size);
			cout << "result_sq: " << endl;
			printArray(result_sq, size);
			cout << "result_pl: " << endl;
			printArray(result_pl, size);

		}

		//check
		bool checkResult = verifyVectorResult(result_sq, result_pl, size, debug);		
		//-----------------------------------------------------
		// STEP 10: Release OpenCL resources
		//-----------------------------------------------------

		// Free OpenCL resources
		clReleaseKernel(mulKernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(cmdQueue);
		clReleaseMemObject(bufferVectorIn);
		clReleaseMemObject(bufferMatrixIn);
		clReleaseMemObject(bufferVectorOut);
		clReleaseContext(context);

		//Free up memory and close files
		free(vector);
		free(matrix);
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

