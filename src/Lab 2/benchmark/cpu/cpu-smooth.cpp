#include "cpu-kernels.h"

using namespace std;

void triangularSmooth(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height, const float *filter, ResultContainer *result, double cpu_frequency) {
	auto t_preprocessing = now();
	auto t_init = t_preprocessing;
	auto t_kernel = t_preprocessing;
	// Kernel
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
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
	// /Kernel
	auto t_cleanup = now();
	auto t_postprocessing = t_cleanup;
	auto t_end = t_cleanup;

	*result = ResultContainer(t_preprocessing, t_init, t_kernel, t_cleanup, t_postprocessing, t_end, cpu_frequency);
}