#include "cpu-kernels.h"

using namespace std;

void triangularSmooth(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height, const float *filter, double cpu_frequency) {

	auto t1 = now();
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
	auto t2 = now();

	cout << fixed << setprecision(6);
	double time_elapsed = diffToNanoseconds(t1, t2, cpu_frequency);
	cout << "triangularSmooth (cpu): \t" << time_elapsed << " nanoseconds." << endl;
}