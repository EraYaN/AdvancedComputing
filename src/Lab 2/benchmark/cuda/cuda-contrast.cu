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
__global__ void contrast1DCudaKernel(unsigned char *grayImage, const int width, const int height, const int min, const int max, const float diff) {
	int x = blockIdx.x * blockDim.x + threadIdx.x; // width
	int y = blockIdx.y * blockDim.y + threadIdx.y; // height

	if (x < width && y < height) {
		unsigned char pixel = grayImage[(y * width) + x];

		if (pixel < min) {
			pixel = 0;
		} else if (pixel > max) {
			pixel = 255;
		} else {
			pixel = static_cast<unsigned char>(255.0f * (pixel - min) / diff);
		}

		grayImage[(y * width) + x] = pixel;
	}
}


void contrast1DCuda(unsigned char *grayImage, const int width, const int height,
	unsigned int *histogram, const unsigned int HISTOGRAM_SIZE,
	const unsigned int CONTRAST_THRESHOLD) {
	unsigned int i = 0;
	NSTimer kernelTime = NSTimer("kernelTime", false, false);

	while ((i < HISTOGRAM_SIZE) && (histogram[i] < CONTRAST_THRESHOLD)) {
		i++;
	}
	unsigned int min = i;

	i = HISTOGRAM_SIZE - 1;
	while ((i > min) && (histogram[i] < CONTRAST_THRESHOLD)) {
		i--;
	}
	unsigned int max = i;
	float diff = max - min;

	kernelTime.start();
	// Kernel

	// specify thread and block dimensions
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

	// allocate GPU memory
	unsigned char *dev_a;
	int size = width * height;

	cudaMalloc((void**)&dev_a, size * (sizeof(unsigned char)));

	// copy grayImage to GPU memory
	cudaMemcpy(dev_a, grayImage, size * (sizeof(unsigned char)), cudaMemcpyHostToDevice);

	// execute actual function
	contrast1DCudaKernel << <numBlocks, threadsPerBlock >> > (dev_a, width, height, min, max, diff);

	// copy result from GPU memory to grayImage
	cudaMemcpy(grayImage, dev_a, size * (sizeof(unsigned char)), cudaMemcpyDeviceToHost);

	// free memory
	cudaFree(dev_a);

	// /Kernel
	kernelTime.stop();

	cout << fixed << setprecision(6);
	cout << "contrast1D (cpu): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
}
