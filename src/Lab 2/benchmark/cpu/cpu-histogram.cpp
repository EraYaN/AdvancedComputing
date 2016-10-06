#include "cpu-kernels.h"

using namespace std;

void histogram1D(unsigned char *grayImage, unsigned char *histogramImage, const int width, const int height, unsigned int *histogram, const unsigned int histogramSize, const unsigned int barWidth, ResultContainer *result, double cpu_frequency) {
	auto t_preprocessing = now();
	unsigned int max = 0;

	memset(reinterpret_cast<void *>(histogram), 0, histogramSize * sizeof(unsigned int));

	auto t_init = now();
	auto t_kernel = t_init;
	// Kernel
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			histogram[static_cast<unsigned int>(grayImage[(y * width) + x])] += 1;
		}
	}
	// /Kernel
	auto t_cleanup = now();
	auto t_postprocessing = t_cleanup;	

	for (unsigned int i = 0; i < histogramSize; i++) {
		if (histogram[i] > max) {
			max = histogram[i];
		}
	}

	for (int x = 0; x < histogramSize * barWidth; x += barWidth) {
		unsigned int value = histogramSize - ((histogram[x / barWidth] * histogramSize) / max);

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
