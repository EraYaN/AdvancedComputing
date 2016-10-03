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


void rgb2grayCuda(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height) {
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

	// copy inputImage to GPU memory
	cudaMemcpy(dev_a, inputImage, size * (sizeof(unsigned char)), cudaMemcpyHostToDevice);

	// execute actual function
	rgb2grayCudaKernel<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, height, width);

	// copy result from GPU memory to grayImage
	cudaMemcpy(grayImage, dev_b, size * (sizeof(unsigned char)), cudaMemcpyDeviceToHost);

	// free memory
	cudaFree(dev_a);
	cudaFree(dev_b);

	// /Kernel
	kernelTime.stop();

	cout << fixed << setprecision(6);
	cout << "rgb2gray (cpu): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
}

/////////////////////////////////////
/*
__global__ void histogram1DCudaKernel
{
}
*/

/*
void histogram1DCuda(unsigned char *grayImage, unsigned char *histogramImage, const int width, const int height,unsigned int *histogram, const unsigned int HISTOGRAM_SIZE,const unsigned int BAR_WIDTH)
{
}
*/


/////////////////////////////////////
/*
__global__ void contrast1DKernel
{
}
*/

/*
void contrast1DCuda(unsigned char *grayImage, const int width, const int height,unsigned int *histogram, const unsigned int HISTOGRAM_SIZE,const unsigned int CONTRAST_THRESHOLD)
{
}
*/

/////////////////////////////////////
/*
__global__ void triangularSmoothKernel
{
}
*/

/*
void triangularSmoothCuda(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height,const float *filter)
{
}
*/

