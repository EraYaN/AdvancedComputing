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
void matrix_mult_sq(int size, cl_double *vector_in,
	cl_double *matrix_in, cl_double *vector_out) {
	int rows, cols;
	int j;
	for (cols = 0; cols < size; cols++) {
		vector_out[cols] = 0.0;
		for (j = 0, rows = 0; rows < size; j++, rows++)
			vector_out[cols] += vector_in[j] * matrix_in[rows*size + cols];
	}
}

/*****************************************************
the following function generates a "size"-element vector
and a "size x size" matrix
 ****************************************************/
void matrix_vector_gen(cl_int size, cl_double *matrix, cl_double *vector) {
	int i;
	for (i = 0; i < size; i++)
		vector[i] = ((double)rand()) / 65535.0;
	for (i = 0; i < size*size; i++)
		matrix[i] = ((double)rand()) / 5307.0;
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
		TCLAP::CmdLine cmd("OpenCL Matrix x Vector Multiplication", ' ', "0.9");

		// Define a value argument and add it to the command line.
		// A value arg defines a flag and a type of value that it expects,
		// such as "-n Bishop".
		TCLAP::ValueArg<unsigned int> threadsArg("t", "threads", "Local workgroup size.", true, 2, "unsigned int");
		TCLAP::ValueArg<unsigned int> datasizeArg("s", "data_size", "Data size.", true, 2, "unsigned int");
		TCLAP::ValuesConstraint<int> variantConstraint(variants);
		TCLAP::ValueArg<int> variantArg("v", "variant", "Variant ID to run.", false, (int)base, &variantConstraint, false);
		TCLAP::SwitchArg debugArg("d", "debug", "Enable debug mode, verbose output.", false);
		TCLAP::SwitchArg interactiveArg("i", "interactive", "Enable interactive mode.", false);

		// Add the argument nameArg to the CmdLine object. The CmdLine object
		// uses this Arg to parse the command line.
		cmd.add(threadsArg);
		cmd.add(datasizeArg);
		cmd.add(variantArg);
		cmd.add(debugArg);
		cmd.add(interactiveArg);

		// Parse the argv array.
		cmd.parse(argc, argv);

		// Get the value parsed by each arg.
		cl_int size = (cl_int)datasizeArg.getValue();
		cl_int localSize = (cl_int)threadsArg.getValue();
		Variant variant = (Variant)variantArg.getValue();
		bool debug = debugArg.getValue();
		bool interactive = interactiveArg.getValue();

		if ((size == 0) || (localSize == 0)) {
			cerr << "Incorrect arguments, make sure both arguments are integers greater than zero." << endl;
			if (interactive) wait_for_input();
			exit(-1);
		} else if ((size % localSize) != 0) {
			cerr << "size (" << size << ") should be a multiple of localSize (" << localSize << ")" << endl;
			if (interactive) wait_for_input();
			exit(-1);
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

		cl_double *vector = (double *)malloc(sizeof(cl_double)*size);
		cl_double *matrix = (double *)malloc(sizeof(cl_double)*size*size);
		cl_double *result_sq = (double *)malloc(sizeof(cl_double)*size);
		cl_double *result_pl = (double *)malloc(sizeof(cl_double)*size);
		matrix_vector_gen(size, matrix, vector);

		double time_sq;
		double time_opencl;


		cl_event mulDone;

		time_sq = omp_get_wtime();
		matrix_mult_sq(size, vector, matrix, result_sq);
		time_sq = omp_get_wtime() - time_sq;

		cl_int status;

		time_opencl = omp_get_wtime();


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
			exit(-1);
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
			exit(-1);
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
			exit(-1);
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
			exit(-1);
		}

		//-----------------------------------------------------
		// STEP 5: Create device buffers, images and copy data to buffers
		//-----------------------------------------------------
		cl_mem bufferVectorIn, bufferMatrixIn, bufferVectorOut;


		bufferVectorIn = clCreateBuffer(
			context,
			CL_MEM_READ_ONLY,
			size * sizeof(cl_double),
			NULL,
			&status);

		if (status != CL_SUCCESS) {
			cerr << "error in step 5, creating buffer for bufferA: " << getErrorString(status) << endl;
			if (interactive) wait_for_input();
			exit(-1);
		}

		bufferMatrixIn = clCreateBuffer(
			context,
			CL_MEM_READ_ONLY,
			size*size * sizeof(cl_double),
			NULL,
			&status);

		if (status != CL_SUCCESS) {
			cerr << "error in step 5, creating buffer for bufferB: " << getErrorString(status) << endl;
			if (interactive) wait_for_input();
			exit(-1);
		}

		bufferVectorOut = clCreateBuffer(
			context,
			CL_MEM_WRITE_ONLY,
			size*size * sizeof(cl_double),
			NULL,
			&status);

		if (status != CL_SUCCESS) {
			cerr << "error in step 5, creating buffer for bufferC: " << getErrorString(status) << endl;
			if (interactive) wait_for_input();
			exit(-1);
		}

		status = clEnqueueWriteBuffer(
			cmdQueue,
			bufferVectorIn,
			CL_FALSE,
			0,
			size * sizeof(cl_double),
			vector,
			0,
			NULL,
			NULL);

		status |= clEnqueueWriteBuffer(
			cmdQueue,
			bufferMatrixIn,
			CL_FALSE,
			0,
			size*size * sizeof(cl_double),
			matrix,
			0,
			NULL,
			NULL);

		if (status != CL_SUCCESS) {
			cerr << "error in step 5, writing data: " << getErrorString(status) << endl;
			if (interactive) wait_for_input();
			exit(-1);
		}

		char *mulFileName;
		mulFileName = "vectorMatrixMul.cl";
		/*FILE *mulFile;
		fopen_s(&mulFile, mulFileName, "r");
		if (mulFile == NULL) {
			cerr << "cannot open .cl file" << endl;
			cerr << "current path: " << mulFileName << endl;
			if (interactive) wait_for_input();
			exit(-1);
		}
		fseek(mulFile, 0, SEEK_END);
		size_t mulSize = ftell(mulFile);
		rewind(mulFile);

		// read kernel source into buffer
		mulBuffer = (char*)malloc(mulSize + 1);
		mulBuffer[mulSize] = '\0';
		fread(mulBuffer, sizeof(char), mulSize, mulFile);
		fclose(mulFile);*/

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
			exit(-1);
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
			exit(-1);
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
			exit(-1);
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
			exit(-1);
		}

		clEnqueueReadBuffer(
			cmdQueue,
			bufferVectorOut,
			CL_TRUE,
			0,
			size * sizeof(cl_double),
			result_pl,
			1,
			&mulDone,
			NULL);


		if (status != CL_SUCCESS) {
			cerr << "error in reading data: " << getErrorString(status) << endl;
			exit(-1);
		}

		time_opencl = omp_get_wtime() - time_opencl;

		if (debug) {
			printf("DAT:%d\n", size);
			printf("THD:%d\n", localSize);
			//printf("PRC:%d\n", omp_get_num_procs());
		}
		printf("SEQ:%.14f\n", time_sq);
		printf("VAR:%.14f\n", time_opencl);


		//check
		int i;
		for (i = 0; i < size; i++) {
			if ((int)result_sq[i] != (int)result_pl[i]) {
				if (debug) {
					cout << "Wrong value \"" << result_sq[i] << "\" and \"" << result_pl[i] << "\" at position " << i << "." << endl;
				}
				if (interactive) {
					wait_for_input();
				}
				return 2;
			}
		}
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

		return EXIT_SUCCESS;
	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{
		cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
		return -1;
	}
}

