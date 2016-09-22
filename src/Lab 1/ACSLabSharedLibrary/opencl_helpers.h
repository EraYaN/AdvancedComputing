#pragma once
#include <time.h>
#include <iostream>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/*** TYPEDEFS AND STRUCTS***/
typedef unsigned long long timestamp_t;

const char *getErrorString(cl_int error);
const char *getBuildStatusString(cl_build_status status);
static timestamp_t get_timestamp();
void printCLBuildOutput(cl_program program, const cl_device_id* device_id);
