#include <Timer.hpp>
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include "../checkCudaCall.h"

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

/////////////////////////////////////
__global__ void triangularSmoothCudaKernel(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height,	const float *filter)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x; // width
	int y = blockIdx.y * blockDim.y + threadIdx.y; // height

	if (x < width && y < height) {
		unsigned int filterItem = 0;
		float filterSum = 0.0f;
		float smoothPix = 0.0f;

		for (int fy = y - 2; fy < y + 3; fy++) {
			for (int fx = x - 2; fx < x + 3; fx++) {
				if (((fy < 0) || (fy >= height)) || ((fx < 0) || (fx >= width))) {
					filterItem++;
					continue;
				}

				smoothPix += grayImage[(fy * width) + fx] * filter[filterItem];
				filterSum += filter[filterItem];
				filterItem++;
			}
		}

		smoothPix /= filterSum;
		smoothImage[(y * width) + x] = static_cast<unsigned char>(smoothPix);
	}
}

void triangularSmoothCuda(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height,	const float *filter) {
	NSTimer kernelTime = NSTimer("kernelTime", false, false);

	kernelTime.start();
	// Kernel

	// specify thread and block dimensions
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

	// allocate GPU memory
	unsigned char *dev_a, *dev_b;
	int size = width * height;

	cudaMalloc((void**)&dev_a, size * (sizeof(unsigned char)));
	cudaMalloc((void**)&dev_b, size * (sizeof(unsigned char)));

	// copy grayImage to GPU memory
	cudaMemcpy(dev_a, grayImage, size * (sizeof(unsigned char)), cudaMemcpyHostToDevice);

	// execute actual function
	triangularSmoothCudaKernel << <numBlocks, threadsPerBlock >> > (dev_a, dev_b, width, height, filter);

	// copy result from GPU memory to grayImage
	cudaMemcpy(smoothImage, dev_b, size * (sizeof(unsigned char)), cudaMemcpyDeviceToHost);

	// free memory
	cudaFree(dev_a);
	cudaFree(dev_b);

	// /Kernel
	kernelTime.stop();

	cout << fixed << setprecision(6);
	cout << "triangularSmooth (cpu): \t" << kernelTime.getElapsed() << " seconds." << endl;
}