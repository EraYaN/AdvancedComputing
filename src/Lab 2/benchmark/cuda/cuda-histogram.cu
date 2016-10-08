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

__global__ void histogram1DCudaKernelShared(unsigned char *grayImage, unsigned int *histogram, const int width, const int height) {
	/*__shared__ unsigned int temp[256];
	temp[threadIdx.x] = 0;
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	while (i < width*height) {
		atomicAdd(&histogram[static_cast<unsigned int>(grayImage[i])], 1);
		i += offset;
	}
	__syncthreads();

	atomicAdd(&(histogram[threadIdx.x]), temp[threadIdx.x]);*/
	// current pixel coordinates
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		// grid dimensions
		int nx = blockDim.x * gridDim.x;
		int ny = blockDim.y * gridDim.y;

		// linear thread index within 2D block
		int t = threadIdx.x + threadIdx.y * blockDim.x;

		// total threads in 2D block
		int nt = blockDim.x * blockDim.y;

		// initialize shared memory
		extern __shared__ unsigned int temp[];

		memset(temp, 0, 256*sizeof(unsigned int));
		/*for (int i = t; i < 256; i += nt) {
			temp[i] = 0;
		}*/

		__syncthreads();
		// build per-block histogram in shared memory
		for (int col = x; col < width; col += nx) {
			for (int row = y; row < height; row += ny) {
				unsigned int i = static_cast<unsigned int>(grayImage[(y * width) + x]);
				atomicAdd(&temp[i], 1);
			}
		}
		__syncthreads();

		// merge per-block histograms to global memory
		for (int i = t; i < 256; i += nt) {
			atomicAdd(&histogram[i], temp[i]);
		}
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

	//checkCudaCall(cudaHostGetDevicePointer(&dev_a, grayImage, 0));
	//checkCudaCall(cudaHostGetDevicePointer(&dev_b, histogram, 0));
	checkCudaCall(cudaMalloc((void **)&dev_a, width*height * sizeof(unsigned char)));
	checkCudaCall(cudaMalloc((void **)&dev_b, histogramSize * sizeof(unsigned int)));
	
	//cudaMemset(dev_b, 0, histogramSize * sizeof(unsigned int));
	cudaMemcpy(dev_a, grayImage, width*height * sizeof(unsigned char), cudaMemcpyHostToDevice);

	auto t_kernel = now();
	// execute actual function
	histogram1DCudaKernel << <numBlocks, threadsPerBlock>> > (dev_a, dev_b, width, height);
	//histogram1DCudaKernelShared <<<numBlocks, threadsPerBlock, histogramSize*sizeof(unsigned int)>>> (dev_a, dev_b, width, height);
	//checkCudaCall(cudaThreadSynchronize());
	auto t_cleanup = now();

	cudaMemcpy(histogram, dev_b, histogramSize * (sizeof(unsigned int)), cudaMemcpyDeviceToHost);
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
