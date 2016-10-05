#include "cpu-kernels.h"

using namespace std;

void rgb2gray(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height, double cpu_frequency) {

	auto t1 = now();
	// Kernel
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float grayPix = 0.0f;
			float r = static_cast<float>(inputImage[(y * width) + x]);
			float g = static_cast<float>(inputImage[(width * height) + (y * width) + x]);
			float b = static_cast<float>(inputImage[(2 * width * height) + (y * width) + x]);

			grayPix = (0.3f * r) + (0.59f * g) + (0.11f * b);

			grayImage[(y * width) + x] = static_cast<unsigned char>(grayPix);
		}
	}
	// /Kernel
	auto t2 = now();

	cout << fixed << setprecision(6);
	double time_elapsed = diffToNanoseconds(t1, t2, cpu_frequency);
	cout << "rgb2gray (cpu): \t\t" << time_elapsed << " nanoseconds." << endl;
}


