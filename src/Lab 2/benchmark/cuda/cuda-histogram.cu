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
__global__ void histogram1DCudaKernel(unsigned char *grayImage, unsigned int *histogram, const int width, const int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x; // width
	int y = blockIdx.y * blockDim.y + threadIdx.y; // height

	if (x < width && y < height) {
		histogram[static_cast<unsigned int>(grayImage[(y * width) + x])] += 1;
	}
}

void histogram1DCuda(unsigned char *grayImage, unsigned char *histogramImage, const int width, const int height,
	unsigned int *histogram, const unsigned int HISTOGRAM_SIZE,
	const unsigned int BAR_WIDTH) {
	unsigned int max = 0;
	NSTimer kernelTime = NSTimer("kernelTime", false, false);

	memset(reinterpret_cast<void *>(histogram), 0, HISTOGRAM_SIZE * sizeof(unsigned int));

	kernelTime.start();
	// Kernel

	// specify thread and block dimensions
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

	// allocate GPU memory
	unsigned char *dev_a;
	unsigned int *dev_b;
	int size = width * height;

	cudaMalloc((void**)&dev_a, size * (sizeof(unsigned char)));
	cudaMalloc((void**)&dev_b, HISTOGRAM_SIZE * (sizeof(unsigned int)));

	// copy grayImage to GPU memory
	cudaMemcpy(dev_a, grayImage, size * (sizeof(unsigned char)), cudaMemcpyHostToDevice);

	// execute actual function
	histogram1DCudaKernel << <numBlocks, threadsPerBlock >> > (dev_a, dev_b, width, height);

	// copy result from GPU memory to histogramImage
	cudaMemcpy(histogram, dev_b, size * (sizeof(unsigned int)), cudaMemcpyDeviceToHost);

	// free memory
	cudaFree(dev_a);
	cudaFree(dev_b);

	// /Kernel
	kernelTime.stop();

	for (unsigned int i = 0; i < HISTOGRAM_SIZE; i++) {
		if (histogram[i] > max) {
			max = histogram[i];
		}
	}

	for (int x = 0; x < HISTOGRAM_SIZE * BAR_WIDTH; x += BAR_WIDTH) {
		unsigned int value = HISTOGRAM_SIZE - ((histogram[x / BAR_WIDTH] * HISTOGRAM_SIZE) / max);

		for (unsigned int y = 0; y < value; y++) {
			for (unsigned int i = 0; i < BAR_WIDTH; i++) {
				histogramImage[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 0;
			}
		}
		for (unsigned int y = value; y < HISTOGRAM_SIZE; y++) {
			for (unsigned int i = 0; i < BAR_WIDTH; i++) {
				histogramImage[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 255;
			}
		}
	}

	cout << fixed << setprecision(6);
	cout << "histogram1D (cpu): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
}
