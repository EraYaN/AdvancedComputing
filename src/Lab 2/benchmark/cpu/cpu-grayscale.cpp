#include <Timer.hpp>
#include <iostream>
#include <iomanip>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

void rgb2gray(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height) {
	NSTimer kernelTime = NSTimer("kernelTime", false, false);

	kernelTime.start();
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
	kernelTime.stop();

	cout << fixed << setprecision(6);
	cout << "rgb2gray (cpu): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
}


