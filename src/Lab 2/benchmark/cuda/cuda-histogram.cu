#include "cuda-kernels.h"

using namespace std;

/////////////////////////////////////
__global__ void histogram1DCudaKernel(unsigned char *grayImage, unsigned int *histogram, const int width, const int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x; // width
	int y = blockIdx.y * blockDim.y + threadIdx.y; // height

	if (x < width && y < height) {
		atomicAdd(&histogram[static_cast<unsigned int>(grayImage[(y * width) + x])], (unsigned int)1);
	}
}

void histogram1DCuda(unsigned char *grayImage, unsigned char *histogramImage, const int width, const int height, unsigned int *histogram, const unsigned int histogramSize, const unsigned int barWidth, ResultContainer *result, double cpu_frequency) {
	auto t_preprocessing = now();
	unsigned int max = 0;

	memset(reinterpret_cast<void *>(histogram), 0, histogramSize * sizeof(unsigned int));

	auto t_init = now();
	// Kernel

	// specify thread and block dimensions
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

	// allocate GPU memory
	unsigned char *dev_a;
	unsigned int *dev_b;

	checkCudaCall(cudaHostGetDevicePointer(&dev_a, grayImage, 0));
	checkCudaCall(cudaHostGetDevicePointer(&dev_b, histogram, 0));

	auto t_kernel = now();
	// execute actual function
	histogram1DCudaKernel<<<numBlocks, threadsPerBlock>>> (dev_a, dev_b, width, height);
	checkCudaCall(cudaThreadSynchronize());
	auto t_cleanup = now();

	// /Kernel
	auto t_postprocessing = now();

	for (unsigned int i = 0; i < histogramSize; i++) {
		if (histogram[i] > max) {
			max = histogram[i];
		}
	}

	for (int x = 0; x < histogramSize * barWidth; x += barWidth) {
		unsigned int value = 0;
		if (max > 0)
			value = histogramSize - ((histogram[x / barWidth] * histogramSize) / max);

		for (unsigned int y = 0; y < value; y++) {
			for (unsigned int i = 0; i < barWidth; i++) {
				histogramImage[(y * histogramSize * barWidth) + x + i] = 0;
			}
		}
		for (unsigned int y = value; y < histogramSize; y++) {
			for (unsigned int i = 0; i < barWidth; i++) {
				histogramImage[(y * histogramSize * barWidth) + x + i] = 255;
			}
		}
	}
	auto t_end = now();

	*result = ResultContainer(t_preprocessing, t_init, t_kernel, t_cleanup, t_postprocessing, t_end, cpu_frequency);


}
