#pragma once
#include "exit_codes.h"
#include "interactive_tools.h"
/* Utility function/macro, used to do error checking.
Use this function/macro like this:
checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));
And to check the result of a kernel invocation:
checkCudaCall(cudaGetLastError());
*/
#define checkCudaCall(result) {                                     \
    if (result != cudaSuccess){                                     \
        cerr << "cuda error: " << cudaGetErrorString(result);       \
        cerr << " in " << __FILE__ << " at line "<< __LINE__<<endl; \
		wait_for_input();											\
        exit(EXIT_CUDAERROR);                                       \
    }                                                               \
}
