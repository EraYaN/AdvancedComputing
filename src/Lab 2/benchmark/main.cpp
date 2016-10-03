#define cimg_use_jpeg
#include <CImg.h>
#include <cuda.h>
#include <Timer.hpp>
#include <iostream>
#include <iomanip>
//#include <cstring>
#include <string>
#include <algorithm>
#include <tclap/CmdLine.h>
#include <tclap/UnlabeledValueArg.h>
#include "exit_codes.h"
#include "interactive_tools.h"
#include "string_format.h"

using cimg_library::CImg;
using LOFAR::NSTimer;
using namespace std;

// Constants

const unsigned int HISTOGRAM_SIZE = 256;
const unsigned int BAR_WIDTH = 4;
const unsigned int CONTRAST_THRESHOLD = 80;
const float filter[] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
						1.0f, 2.0f, 2.0f, 2.0f, 1.0f,
						1.0f, 2.0f, 3.0f, 2.0f, 1.0f,
						1.0f, 2.0f, 2.0f, 2.0f, 1.0f,
						1.0f, 1.0f, 1.0f, 1.0f, 1.0f };

extern void rgb2gray(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height);
extern void rgb2grayCuda(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height);

extern void histogram1D(unsigned char *grayImage, unsigned char *histogramImage, const int width, const int height, unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, const unsigned int BAR_WIDTH);
//extern void histogram1DCuda

extern void contrast1D(unsigned char *grayImage, const int width, const int height, unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, const unsigned int CONTRAST_THRESHOLD);
//extern void contrast1DCuda

extern void triangularSmooth(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height, const float *filter);
//extern void triangularSmoothCuda


int main(int argc, char *argv[]) {
	try {
		// Define the command line object, and insert a message
		// that describes the program. The "Command description message"
		// is printed last in the help text. The second argument is the
		// delimiter (usually space) and the last one is the version number.
		// The CmdLine object parses the argv array based on the Arg objects
		// that it contains.
		TCLAP::CmdLine cmd("Image Comparison Tool", ' ', "1.0");

		// Define a value argument and add it to the command line.
		// A value arg defines a flag and a type of value that it expects,
		// such as "-n Bishop".
		TCLAP::UnlabeledValueArg<string> imageArg("input", "The image to be processed.", true, "", "infile");
		TCLAP::UnlabeledValueArg<string> outputArg("output", "The output images prefix.", false, "./out-cuda", "outfile");
		TCLAP::ValueArg<unsigned int> histogramSizeArg("hs", "histogram-size", "The histogram size.", false, 256, "size");
		TCLAP::ValueArg<unsigned int> histogramBarWidthArg("hb", "histogram-bar-width", "The histogram bar width.", false, 4, "pixels");
		TCLAP::ValueArg<unsigned int> thresholdArg("t", "contrast-threshold", "The contrast threshold.", false, 80, "pixels");
		TCLAP::SwitchArg displayArg("di", "display-images", "Enable image display.", false);
		TCLAP::SwitchArg saveArg("si", "save-images", "Enable image save.", false);
		TCLAP::SwitchArg debugArg("d", "debug", "Enable debug mode, verbose output.", false);
		TCLAP::SwitchArg interactiveArg("i", "interactive", "Enable interactive mode.", false);

		// Add the argument nameArg to the CmdLine object. The CmdLine object
		// uses this Arg to parse the command line.
		cmd.add(imageArg);
		cmd.add(outputArg);
		cmd.add(histogramSizeArg);
		cmd.add(histogramBarWidthArg);
		cmd.add(thresholdArg);
		cmd.add(displayArg);
		cmd.add(saveArg);
		cmd.add(debugArg);
		cmd.add(interactiveArg);

		// Parse the argv array.
		cmd.parse(argc, argv);

		// Get the value parsed by each arg.
		CImg< unsigned char > inputImage = CImg< unsigned char >(imageArg.getValue().c_str());
		string output = outputArg.getValue() + "-{0}.bmp";
		bool displayImages = displayArg.getValue();
		bool saveAllImages = saveArg.getValue();
		unsigned int histogramSize = histogramSizeArg.getValue();
		unsigned int barWidth = histogramBarWidthArg.getValue();
		unsigned int contrastThreshold = thresholdArg.getValue();
		bool debug = debugArg.getValue();
		bool interactive = interactiveArg.getValue();

		// Load the input image
		if (displayImages) {
			inputImage.display("Input Image");
		}
		if (inputImage.spectrum() != 3) {
			if (interactive) cerr << "The input must be a color image." << endl;
			return EXIT_BADINPUT;
		}

		// Convert the input image to grayscale
		CImg< unsigned char > grayImage = CImg< unsigned char >(inputImage.width(), inputImage.height(), 1, 1);

		//rgb2gray(inputImage.data(), grayImage.data(), inputImage.width(), inputImage.height());
		rgb2grayCuda(inputImage.data(), grayImage.data(), inputImage.width(), inputImage.height());

		if (displayImages) {
			grayImage.display("Grayscale Image");
		}
		if (saveAllImages) {
			grayImage.save(string_format(output, "grayscale").c_str());
		}

		// Compute 1D histogram
		CImg< unsigned char > histogramImage = CImg< unsigned char >(barWidth * histogramSize, histogramSize, 1, 1);
		unsigned int *histogram = new unsigned int[histogramSize];

		histogram1D(grayImage.data(), histogramImage.data(), grayImage.width(), grayImage.height(), histogram, histogramSize, barWidth);
		//histogram1DCuda

		if (displayImages) {
			histogramImage.display("Histogram");
		}
		if (saveAllImages) {
			histogramImage.save(string_format(output, "histogram").c_str());
		}

		// Contrast enhancement
		contrast1D(grayImage.data(), grayImage.width(), grayImage.height(), histogram, histogramSize, contrastThreshold);
		//contrast1DCuda

		if (displayImages) {
			grayImage.display("Contrast Enhanced Image");
		}
		if (saveAllImages) {
			grayImage.save(string_format(output, "contrast").c_str());
		}

		delete[] histogram;

		// Triangular smooth (convolution)
		CImg< unsigned char > smoothImage = CImg< unsigned char >(grayImage.width(), grayImage.height(), 1, 1);

		triangularSmooth(grayImage.data(), smoothImage.data(), grayImage.width(), grayImage.height(), filter);
		//triangularSmoothCuda

		if (displayImages) {
			smoothImage.display("Smooth Image");
		}

		if (saveAllImages) {
			smoothImage.save(string_format(output, "smooth").c_str());
		}

		if (interactive) {
			wait_for_input();
		}

		return EXIT_SUCCESS;
	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{
		cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
		return EXIT_BADARGUMENT;
	}
}

