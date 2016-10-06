#include "cpu-kernels.h"

using namespace std;

void rgb2gray(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height, ResultContainer *result, double cpu_frequency) {

	auto t_preprocessing = now();
	auto t_init = t_preprocessing;
	auto t_kernel = t_init;
	// Kernel
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float r = static_cast<float>(inputImage[(y * width) + x]);
			float g = static_cast<float>(inputImage[(width * height) + (y * width) + x]);
			float b = static_cast<float>(inputImage[(2 * width * height) + (y * width) + x]);

			float grayPix = (0.3f * r) + (0.59f * g) + (0.11f * b);

			grayImage[(y * width) + x] = static_cast<unsigned char>(grayPix);
		}
	}
	// /Kernel
	auto t_cleanup = now();
	auto t_postprocessing = t_cleanup;
	auto t_end = t_cleanup;

	*result = ResultContainer(t_preprocessing, t_init, t_kernel, t_cleanup, t_postprocessing, t_end, cpu_frequency);
	return;
}


