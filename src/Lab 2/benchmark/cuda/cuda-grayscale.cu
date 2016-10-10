#include "cuda-kernels.h"

using namespace std;

__global__ void rgb2grayCudaKernel(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x; // width
	int y = blockIdx.y * blockDim.y + threadIdx.y; // height

	if (x < width && y < height) {
		float r = static_cast<float>(inputImage[(y * width) + x]);
		float g = static_cast<float>(inputImage[(width * height) + (y * width) + x]);
		float b = static_cast<float>(inputImage[(2 * width * height) + (y * width) + x]);

		float grayPix = (0.3f * r) + (0.59f * g) + (0.11f * b);

		grayImage[((y * width) + x)] = static_cast<unsigned char>(grayPix);
	}
}

void rgb2grayCuda(unsigned char *inputImage, unsigned char *grayImage, unsigned char *dev_grayImage, const int width, const int height, ResultContainer *result, double cpu_frequency) {
	auto t_preprocessing = now();
	auto t_init = t_preprocessing;
	// Kernel

	// specify thread and block dimensions
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(ceil((double)width / threadsPerBlock.x), ceil((double)height / threadsPerBlock.y));

	// allocate GPU memory
	unsigned char *dev_inputImage;

	//checkCudaCall(cudaHostGetDevicePointer(&dev_inputImage, inputImage, 0));

	checkCudaCall(cudaMalloc(&dev_inputImage, 3*width*height * sizeof(unsigned char)));

	checkCudaCall(cudaMemcpy(dev_inputImage, inputImage, 3 * width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));

	auto t_kernel = now();
	// execute actual function
	rgb2grayCudaKernel<<<numBlocks, threadsPerBlock>>>(dev_inputImage, dev_grayImage, width, height);
	//checkCudaCall(cudaThreadSynchronize());
	auto t_cleanup = now();

	checkCudaCall(cudaMemcpy(grayImage, dev_grayImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	checkCudaCall(cudaFree(dev_inputImage));

	// /Kernel
	auto t_postprocessing = now();
	auto t_end = t_postprocessing;

	*result = ResultContainer(t_preprocessing, t_init, t_kernel, t_cleanup, t_postprocessing, t_end, cpu_frequency);
}

