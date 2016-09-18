#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

int main(void) {
	cl_int err;

	int i, j;
	char* value;
	size_t valueSize;
	cl_uint platformCount;
	cl_platform_id* platforms;
	cl_uint deviceCount;
	cl_device_id* devices;
	cl_uint maxComputeUnits;
	cl_ulong globalMemorySize;
	cl_ulong localMemorySize;
	cl_uint maxClockFrequency;
	cl_uint maxSamplers;
	cl_uint maxWorkGroupSize;

	// get all platforms
	err = clGetPlatformIDs(0, NULL, &platformCount);
	if (CL_SUCCESS == err) {
		platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
		clGetPlatformIDs(platformCount, platforms, NULL);

		for (i = 0; i < platformCount; i++) {
			printf("Platform %d\n", i+1);
			// get all devices
			clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
			devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);
			clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

			// for each device print critical attributes
			for (j = 0; j < deviceCount; j++) {

				// print device name
				clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
				value = (char*)malloc(valueSize);
				clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
				printf(" %d.%d. Device: %s\n", i + 1, j + 1, value);
				free(value);

				// print device vendor
				clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, 0, NULL, &valueSize);
				value = (char*)malloc(valueSize);
				clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, valueSize, value, NULL);
				printf(" %d.%d.%d Vendor: %s\n", i + 1, j + 1, 0, value);
				free(value);

				// print hardware device version
				clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
				value = (char*)malloc(valueSize);
				clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
				printf("  %d.%d.%d Hardware version: %s\n", i + 1, j + 1, 1, value);
				free(value);

				// print software driver version
				clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
				value = (char*)malloc(valueSize);
				clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
				printf("  %d.%d.%d Software version: %s\n", i + 1, j + 1, 2, value);
				free(value);

				// print c version supported by compiler for device
				clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
				value = (char*)malloc(valueSize);
				clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
				printf("  %d.%d.%d OpenCL C version: %s\n", i + 1, j + 1, 3, value);
				free(value);

				// print parallel compute units
				clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
					sizeof(maxComputeUnits), &maxComputeUnits, NULL);
				printf("  %d.%d.%d Parallel compute units: %u\n", i + 1, j + 1, 4, maxComputeUnits);

				// print global memory size
				clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE,
					sizeof(globalMemorySize), &globalMemorySize, NULL);
				printf("  %d.%d.%d Global memory size: %lu MiB\n", i + 1, j + 1, 5, globalMemorySize/1024/1024);

				// print local memory size
				clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE,
					sizeof(localMemorySize), &localMemorySize, NULL);
				printf("  %d.%d.%d Local memory size: %lu KiB\n", i + 1, j + 1, 6, localMemorySize / 1024 );

				// print global memory size
				clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY,
					sizeof(maxClockFrequency), &maxClockFrequency, NULL);
				printf("  %d.%d.%d Max clock frequency: %u MHz\n", i + 1, j + 1, 7, maxClockFrequency);

				// print max samplers
				clGetDeviceInfo(devices[j], CL_DEVICE_MAX_SAMPLERS,
					sizeof(maxSamplers), &maxSamplers, NULL);
				printf("  %d.%d.%d Max samplers: %u\n", i + 1, j + 1, 8, maxSamplers);

				// print max work group size
				clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE,
					sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
				printf("  %d.%d.%d Max work group size: %u\n", i + 1, j + 1, 9, maxWorkGroupSize);

				printf("\n");
			}
			free(devices);
			printf("\n");
		}

		free(platforms);
	} else {
		printf("clGetPlatformIDs threw error code %d", err);
	}
	printf("Press enter to exit.\n");
	getchar();

	return 0;
}