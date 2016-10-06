#include "cpu-kernels.h"

using namespace std;

void contrast1D(unsigned char *grayImage, const int width, const int height, unsigned int *histogram, const unsigned int histogramSize, const unsigned int contrastThreshold, ResultContainer *result, double cpu_frequency) {
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
	auto t_kernel = t_init;
	// Kernel
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
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
	// /Kernel
	auto t_cleanup = now();
	auto t_postprocessing = t_cleanup;
	auto t_end = t_cleanup;

	*result = ResultContainer(t_preprocessing, t_init, t_kernel, t_cleanup, t_postprocessing, t_end, cpu_frequency);
}