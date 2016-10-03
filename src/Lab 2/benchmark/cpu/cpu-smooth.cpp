#include <Timer.hpp>
#include <iostream>
#include <iomanip>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

void triangularSmooth(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height,
	const float *filter) {
	NSTimer kernelTime = NSTimer("kernelTime", false, false);

	kernelTime.start();
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
	kernelTime.stop();

	cout << fixed << setprecision(6);
	cout << "triangularSmooth (cpu): \t" << kernelTime.getElapsed() << " seconds." << endl;
}