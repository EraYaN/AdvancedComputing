#include <Timer.hpp>
#include <iostream>
#include <iomanip>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

void histogram1D(unsigned char *grayImage, unsigned char *histogramImage, const int width, const int height,
	unsigned int *histogram, const unsigned int HISTOGRAM_SIZE,
	const unsigned int BAR_WIDTH) {
	unsigned int max = 0;
	NSTimer kernelTime = NSTimer("kernelTime", false, false);

	memset(reinterpret_cast<void *>(histogram), 0, HISTOGRAM_SIZE * sizeof(unsigned int));

	kernelTime.start();
	// Kernel
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			histogram[static_cast<unsigned int>(grayImage[(y * width) + x])] += 1;
		}
	}
	// /Kernel
	kernelTime.stop();

	for (unsigned int i = 0; i < HISTOGRAM_SIZE; i++) {
		if (histogram[i] > max) {
			max = histogram[i];
		}
	}

	for (int x = 0; x < HISTOGRAM_SIZE * BAR_WIDTH; x += BAR_WIDTH) {
		unsigned int value = HISTOGRAM_SIZE - ((histogram[x / BAR_WIDTH] * HISTOGRAM_SIZE) / max);

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
	cout << "histogram1D (cpu): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
}
