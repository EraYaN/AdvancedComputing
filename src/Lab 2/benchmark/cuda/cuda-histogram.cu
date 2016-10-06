#include "cuda-kernels.h"

using namespace std;

/////////////////////////////////////
__global__ void histogram1DCudaKernel(unsigned char *grayImage, unsigned int *histogram, const int width, const int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x; // width
	int y = blockIdx.y * blockDim.y + threadIdx.y; // height

	if (x < width && y < height) {
		atomicAdd((unsigned int *)histogram[static_cast<unsigned int>(grayImage[(y * width) + x])], (unsigned int)1);
	}
}

void histogram1DCuda(unsigned char *grayImage, unsigned char *histogramImage, const int width, const int height, unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, const unsigned int BAR_WIDTH, double cpu_frequency) {
	unsigned int max = 0;

	memset(reinterpret_cast<void *>(histogram), 0, HISTOGRAM_SIZE * sizeof(unsigned int));

	auto t1 = now();
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
	histogram1DCudaKernel <<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, width, height);

	// copy result from GPU memory to histogramImage
	cudaMemcpy(histogram, dev_b, HISTOGRAM_SIZE * (sizeof(unsigned int)), cudaMemcpyDeviceToHost);

	// free memory
	cudaFree(dev_a);
	cudaFree(dev_b);

	// /Kernel
	auto t2 = now();

	for (unsigned int i = 0; i < HISTOGRAM_SIZE; i++) {
		if (histogram[i] > max) {
			max = histogram[i];
		}
	}

	for (int x = 0; x < HISTOGRAM_SIZE * BAR_WIDTH; x += BAR_WIDTH) {
		unsigned int value = 0;
		if(max>0)
			value = HISTOGRAM_SIZE - ((histogram[x / BAR_WIDTH] * HISTOGRAM_SIZE) / max);

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
	double time_elapsed = diffToNanoseconds(t1, t2, cpu_frequency);
	cout << "histogram1D (cpu): \t\t" << time_elapsed << " nanoseconds." << endl;
}
