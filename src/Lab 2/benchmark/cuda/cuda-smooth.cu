#include "cuda-kernels.h"

using namespace std;

/////////////////////////////////////
__global__ void triangularSmoothCudaKernel(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height, float *filter) {
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

//TODO Implement NPP: http://docs.nvidia.com/cuda/pdf/NPP_Library_Image_Filters.pdf  nppiFilterBox_8u_C1R on page 247
void triangularSmoothCuda(unsigned char *grayImage, unsigned char *dev_grayImage, unsigned char *smoothImage, const int width, const int height, float *filter, ResultContainer *result, double cpu_frequency) {
	auto t_preprocessing = now();
	auto t_init = t_preprocessing;
	// Kernel

	// specify thread and block dimensions
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(ceil((double)width / threadsPerBlock.x), ceil((double)height / threadsPerBlock.y));

	// allocate GPU memory
	unsigned char *dev_smoothImage;
	float *dev_filter;

	//checkCudaCall(cudaHostGetDevicePointer(&dev_smoothImage, smoothImage, 0));
	//checkCudaCall(cudaHostGetDevicePointer(&dev_filter, filter, 0));

	//regular memcpy is much faster than the mapped host memory

	checkCudaCall(cudaMalloc(&dev_smoothImage, width*height * sizeof(unsigned char)));

	checkCudaCall(cudaMalloc(&dev_filter, width*height * sizeof(unsigned char)));
	checkCudaCall(cudaMemcpy(dev_filter, filter, 25 * sizeof(float), cudaMemcpyHostToDevice));

	auto t_kernel = now();
	// execute actual function
	triangularSmoothCudaKernel<<<numBlocks, threadsPerBlock>>>(dev_grayImage, dev_smoothImage, width, height, dev_filter);
	//checkCudaCall(cudaThreadSynchronize());
	auto t_cleanup = now();

	checkCudaCall(cudaMemcpy(smoothImage, dev_smoothImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	checkCudaCall(cudaFree(dev_smoothImage));
	checkCudaCall(cudaFree(dev_filter));

	// /Kernel
	auto t_postprocessing = now();
	auto t_end = t_postprocessing;

	*result = ResultContainer(t_preprocessing, t_init, t_kernel, t_cleanup, t_postprocessing, t_end, cpu_frequency);
}