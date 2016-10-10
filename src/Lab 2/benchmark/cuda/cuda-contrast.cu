#include "cuda-kernels.h"

using namespace std;


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

void contrast1DCuda(unsigned char *grayImage, unsigned char *dev_grayImage, const int width, const int height, unsigned int *histogram, const unsigned int histogramSize, const unsigned int contrastThreshold, ResultContainer *result, double cpu_frequency) {
	auto t_preprocessing = now();
	unsigned int i = 0;

	while ((i < histogramSize) && (histogram[i] < contrastThreshold)) {
		i++;
	}
	unsigned int min = i;

	i = histogramSize - 1;
	while ((i > min) && (histogram[i] < contrastThreshold)) {
		i--;
	}
	unsigned int max = i;
	float diff = max - min;

	auto t_init = now();
	// Kernel

	// specify thread and block dimensions
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(ceil((double)width / threadsPerBlock.x), ceil((double)height / threadsPerBlock.y));
		
	auto t_kernel = now();
	// execute actual function
	contrast1DCudaKernel<<<numBlocks, threadsPerBlock>>>(dev_grayImage, width, height, min, max, diff);
	//checkCudaCall(cudaThreadSynchronize());
	auto t_cleanup = now();

	checkCudaCall(cudaMemcpy(grayImage, dev_grayImage, width*height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	// /Kernel
	auto t_postprocessing = now();
	auto t_end = t_postprocessing;

	*result = ResultContainer(t_preprocessing, t_init, t_kernel, t_cleanup, t_postprocessing, t_end, cpu_frequency);
}