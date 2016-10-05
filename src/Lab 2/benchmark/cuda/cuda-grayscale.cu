#include "cuda-kernels.h"

using namespace std;

__global__ void rgb2grayCudaKernel(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x; // width
	int y = blockIdx.y * blockDim.y + threadIdx.y; // height

	if (x < width && y < height) {
		float grayPix = 0.0f;
		float r = static_cast<float>(inputImage[(y * width) + x]);
		float g = static_cast<float>(inputImage[(width * height) + (y * width) + x]);
		float b = static_cast<float>(inputImage[(2 * width * height) + (y * width) + x]);

		grayPix = (0.3f * r) + (0.59f * g) + (0.11f * b);

		grayImage[(y * width) + x] = static_cast<unsigned char>(grayPix);
	}
}

void rgb2grayCuda(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height, double cpu_frequency) {
	auto t1 = now();
	// Kernel

	// specify thread and block dimensions
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

	// allocate GPU memory
	unsigned char *dev_a, *dev_b;
	int size = width * height;

	cudaMalloc((void**)&dev_a, size * (sizeof(unsigned char)));
	cudaMalloc((void**)&dev_b, size * (sizeof(unsigned char)));

	// copy inputImage to GPU memory
	cudaMemcpy(dev_a, inputImage, size * (sizeof(unsigned char)), cudaMemcpyHostToDevice);

	// execute actual function
	rgb2grayCudaKernel << <numBlocks, threadsPerBlock >> > (dev_a, dev_b, width, height);

	// copy result from GPU memory to grayImage
	cudaMemcpy(grayImage, dev_b, size * (sizeof(unsigned char)), cudaMemcpyDeviceToHost);

	// free memory
	cudaFree(dev_a);
	cudaFree(dev_b);

	// /Kernel
	auto t2 = now();

	cout << fixed << setprecision(6);
	double time_elapsed = diffToNanoseconds(t1, t2, cpu_frequency);
	cout << "rgb2gray (cpu): \t\t" << time_elapsed << " nanoseconds." << endl;
}

