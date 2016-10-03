#include <Timer.hpp>
#include <iostream>
#include <iomanip>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

void contrast1D(unsigned char *grayImage, const int width, const int height,
	unsigned int *histogram, const unsigned int HISTOGRAM_SIZE,
	const unsigned int CONTRAST_THRESHOLD) {
	unsigned int i = 0;
	NSTimer kernelTime = NSTimer("kernelTime", false, false);

	while ((i < HISTOGRAM_SIZE) && (histogram[i] < CONTRAST_THRESHOLD)) {
		i++;
	}
	unsigned int min = i;

	i = HISTOGRAM_SIZE - 1;
	while ((i > min) && (histogram[i] < CONTRAST_THRESHOLD)) {
		i--;
	}
	unsigned int max = i;
	float diff = max - min;

	kernelTime.start();
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
	kernelTime.stop();

	cout << fixed << setprecision(6);
	cout << "contrast1D (cpu): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
}