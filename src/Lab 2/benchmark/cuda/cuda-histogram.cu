#include "cuda-kernels.h"

using namespace std;

/////////////////////////////////////
__global__ void histogram1DCudaKernel(unsigned char *grayImage, unsigned int *histogram, const int width, const int height) {
	// current pixel location
	int x = blockIdx.x * blockDim.x + threadIdx.x; // width
	int y = blockIdx.y * blockDim.y + threadIdx.y; // height

	if (x < width && y < height) {
		atomicAdd(&histogram[static_cast<unsigned int>(grayImage[(y * width) + x])], (unsigned int)1);
	}
}

//Sadly hardcoded histogram size
__global__ void histogram1DCudaKernelShared(unsigned char *grayImage, unsigned int *histogram, const int width, const int height) {
	// pixel coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// grid dimensions
	int nx = blockDim.x * gridDim.x;
	int ny = blockDim.y * gridDim.y;

	// linear thread index within 2D block (start index in block)
	int t = threadIdx.x + threadIdx.y * blockDim.x;

	// total threads in 2D block
	int nt = blockDim.x * blockDim.y;

	// initialize temporary accumulation array in shared memory
	__shared__ unsigned int shared_histogram[256 + 1];
	for (int i = t; i < 256 + 1; i += nt) shared_histogram[i] = 0;
	__syncthreads();

	// process pixels
	// updates our block's partial histogram in shared memory
	for (int col = x; col < width; col += nx)
		for (int row = y; row < height; row += ny) {
			atomicAdd(&shared_histogram[static_cast<unsigned int>(grayImage[row * width + col])], 1);
		}
	__syncthreads();

	// write partial histogram into the global memory
	for (int i = t; i < 256; i += nt) {
		atomicAdd(&histogram[i], shared_histogram[i]);
	}
}

void histogram1DCuda(unsigned char *grayImage, unsigned char *dev_grayImage, unsigned char *histogramImage, const int width, const int height, unsigned int *histogram, const unsigned int histogramSize, const unsigned int barWidth, ResultContainer *result, double cpu_frequency, bool shared) {
	auto t_preprocessing = now();
	unsigned int max = 0;

	auto t_init = now();
	// Kernel

	// specify thread and block dimensions
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(ceil((double)width / threadsPerBlock.x), ceil((double)height / threadsPerBlock.y));

	// allocate GPU memory
	unsigned int *dev_histogram;

	//checkCudaCall(cudaHostGetDevicePointer(&dev_a, grayImage, 0));
	//checkCudaCall(cudaHostGetDevicePointer(&dev_b, histogram, 0));
	checkCudaCall(cudaMalloc((void **)&dev_histogram, histogramSize * sizeof(unsigned int)));

	checkCudaCall(cudaMemset(dev_histogram, 0, histogramSize * sizeof(unsigned int)));

	auto t_kernel = now();
	// execute actual function, with fall back for differing sizes.
	if (histogramSize == 256 && shared) {
		histogram1DCudaKernelShared <<<numBlocks, threadsPerBlock >>> (dev_grayImage, dev_histogram, width, height);
	} else {
		histogram1DCudaKernel <<<numBlocks, threadsPerBlock >>> (dev_grayImage, dev_histogram, width, height);
	}
	//checkCudaCall(cudaThreadSynchronize());
	auto t_cleanup = now();

	checkCudaCall(cudaMemcpy(histogram, dev_histogram, histogramSize * (sizeof(unsigned int)), cudaMemcpyDeviceToHost));
	checkCudaCall(cudaFree(dev_histogram));
	// Kernel

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
