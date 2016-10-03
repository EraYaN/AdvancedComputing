#include <CImg.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <tclap/CmdLine.h>
#include <tclap/UnlabeledValueArg.h>
#include "exit_codes.h"
#include "interactive_tools.h"

using cimg_library::CImg;

using namespace std;


int main(int argc, char *argv[]) {
	long long unsigned int pixelsAboveThreshold = 0;

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
		TCLAP::UnlabeledValueArg<string> image1Arg("image1", "The first image to be compared.", true, "", "infile");
		TCLAP::UnlabeledValueArg<string> image2Arg("image2", "The second image to be compared.", true, "", "infile");
		TCLAP::UnlabeledValueArg<string> outputArg("output", "The output image with the difference.", false, "./diff.bmp", "outfile");
		TCLAP::ValueArg<unsigned int> thresholdArg("t", "threshold", "The difference threshold.", false, 16, "pixels");
		TCLAP::SwitchArg debugArg("d", "debug", "Enable debug mode, verbose output.", false);
		TCLAP::SwitchArg interactiveArg("i", "interactive", "Enable interactive mode.", false);

		// Add the argument nameArg to the CmdLine object. The CmdLine object
		// uses this Arg to parse the command line.
		cmd.add(image1Arg);
		cmd.add(image2Arg);
		cmd.add(outputArg);
		cmd.add(thresholdArg);
		cmd.add(debugArg);
		cmd.add(interactiveArg);

		// Parse the argv array.
		cmd.parse(argc, argv);

		// Get the value parsed by each arg.
		CImg< unsigned char > imageOne = CImg< unsigned char >(image1Arg.getValue().c_str());
		CImg< unsigned char > imageTwo = CImg< unsigned char >(image2Arg.getValue().c_str());
		string output = outputArg.getValue();
		unsigned int threshold = thresholdArg.getValue();
		bool debug = debugArg.getValue();
		bool interactive = interactiveArg.getValue();

		if (imageOne.width() != imageTwo.width()) {
			if(interactive) cerr << "The two images have different width." << endl;
			return EXIT_FAILURE;
		}
		if (imageOne.height() != imageTwo.height()) {
			if (interactive) cerr << "The two images have different height." << endl;
			return EXIT_FAILURE;
		}
		if (imageOne.spectrum() != imageTwo.spectrum()) {
			if (interactive) cerr << "The two images have different spectrum." << endl;
			return EXIT_FAILURE;
		}

		CImg< unsigned char > differenceImage = CImg< unsigned char >(imageOne.width(), imageOne.height(), 1, imageOne.spectrum());

		for (int y = 0; y < differenceImage.height(); y++) {
			for (int x = 0; x < differenceImage.width(); x++) {
				for (int c = 0; c < differenceImage.spectrum(); c++) {
					differenceImage(x, y, 0, c) = abs(imageOne(x, y, 0, c) - imageTwo(x, y, 0, c));

					if (differenceImage(x, y, 0, c) > threshold) {
						pixelsAboveThreshold++;
					}
				}
			}
		}

		differenceImage.save(output.c_str());

		cout << "PXL:" << differenceImage.size() << endl;
		cout << "PTH:" << pixelsAboveThreshold << endl;

		if (interactive) cout << "Pixels above threshold: " << pixelsAboveThreshold << " (" << static_cast<int>((100.0f * pixelsAboveThreshold) / differenceImage.size()) << "%)." << endl;


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

