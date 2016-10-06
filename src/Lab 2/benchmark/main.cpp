#ifdef USE_LIBJPEG
#define cimg_use_jpeg
#endif
#include <CImg.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <tclap/CmdLine.h>
#include <tclap/UnlabeledValueArg.h>
#include "exit_codes.h"
#include "interactive_tools.h"
#include "string_format.h"
#include "timing.h"
#include "cuda_tools.h"
#include "cpu/cpu-kernels.h"
#include "cuda/cuda-kernels.h"
#include "result_container.h"


using cimg_library::CImg;
using namespace std;

static double cpu_frequency = 1;

// Constants

const float filter[] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
						1.0f, 2.0f, 2.0f, 2.0f, 1.0f,
						1.0f, 2.0f, 3.0f, 2.0f, 1.0f,
						1.0f, 2.0f, 2.0f, 2.0f, 1.0f,
						1.0f, 1.0f, 1.0f, 1.0f, 1.0f };


int main(int argc, char *argv[]) {
	try {
		// Define the command line object, and insert a message
		// that describes the program. The "Command description message"
		// is printed last in the help text. The second argument is the
		// delimiter (usually space) and the last one is the version number.
		// The CmdLine object parses the argv array based on the Arg objects
		// that it contains.
		TCLAP::CmdLine cmd("CUDA Image Processing Benchmarking Tool", ' ', "1.0");

		// Define a value argument and add it to the command line.
		// A value arg defines a flag and a type of value that it expects,
		// such as "-n Bishop".
		TCLAP::UnlabeledValueArg<string> imageArg("input", "The image to be processed.", true, "", "infile");
		TCLAP::UnlabeledValueArg<string> outputArg("output", "The output images prefix.", false, "./out-benchmark", "outfile");
		TCLAP::ValueArg<unsigned int> histogramSizeArg("", "histogram-size", "The histogram size.", false, 256, "size");
		TCLAP::ValueArg<unsigned int> histogramBarWidthArg("", "histogram-bar-width", "The histogram bar width.", false, 4, "pixels");
		TCLAP::ValueArg<unsigned int> thresholdArg("", "contrast-threshold", "The contrast threshold.", false, 80, "pixels");
		TCLAP::SwitchArg displayArg("", "display-images", "Enable image display.", false);
		TCLAP::SwitchArg saveArg("", "save-images", "Enable image save.", false);
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
		string output = outputArg.getValue() + "-%s.bmp";
		bool displayImages = displayArg.getValue();
		bool saveAllImages = saveArg.getValue();
		unsigned int histogramSize = histogramSizeArg.getValue();
		unsigned int barWidth = histogramBarWidthArg.getValue();
		unsigned int contrastThreshold = thresholdArg.getValue();
		bool debug = debugArg.getValue();
		bool interactive = interactiveArg.getValue();

#ifdef USE_RDTSC
		cpu_frequency = get_frequency(debug);
#endif
		int nDevices;
		cudaGetDeviceCount(&nDevices);
		cudaDeviceProp prop;
		int biggestDevice = -1;
		int maxPerf = 0;
		cudaDeviceProp biggestDeviceProp;
		for (int i = 0; i < nDevices; i++) {
			cudaGetDeviceProperties(&prop, i);
			int cudaCores = prop.multiProcessorCount*ConvertSMVer2Cores(prop.major, prop.minor);
			if (prop.canMapHostMemory != 1) {
				cerr << "Device " << i << " can not map memory, skipping." << endl;
				continue;
			}
			if (cudaCores*prop.clockRate > maxPerf) {
				maxPerf = cudaCores*prop.clockRate;
				biggestDevice = i;
				biggestDeviceProp = prop;
			}

			if (debug) {

				printf("Device Number: %d\n", i);
				printf("  Device name: %s\n", prop.name);
				printf("  Device Clock Rate: %d MHz\n", prop.clockRate / 1000);
				printf("  Memory Clock Rate: %d MHz\n", prop.memoryClockRate / 1000);
				printf("  Global Memory Size: %d MiB\n", prop.totalGlobalMem / 1024 / 1024);
				printf("  Block registers: %d \n", prop.regsPerBlock);
				printf("  Warp Size: %d \n", prop.warpSize);
				printf("  Thread per block: %d \n", prop.maxThreadsPerBlock);
				printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
				printf("  Peak Memory Bandwidth: %.2f GB/s\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
				printf("  Async Engine count: %d\n", prop.asyncEngineCount);
				printf("  Number of Streaming processors: %d\n", prop.multiProcessorCount);
				printf("  CUDA Cores: %d\n", cudaCores);

				printf("\n");
			}
		}

		if (debug) {
			if (biggestDevice >= 0) {
				cout << "Running benchmarks on " << biggestDeviceProp.name << endl;
				cudaSetDevice(biggestDevice);
				cudaSetDeviceFlags(cudaDeviceMapHost);
			} else {
				cerr << "Could not find proper CUDA device." << endl;
				return EXIT_CUDAERROR;
			}
		}
		//Result Struct
		ResultContainer result;
		//Pinned memory pointers
		unsigned char *inputImagePinned;
		unsigned int *histogramPinned;
		unsigned char *grayImagePinned;
		unsigned char *smoothImagePinned;
		float *filterPinned;

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

		//do both CUDA and Seq.
		rgb2gray(inputImage.data(), grayImage.data(), inputImage.width(), inputImage.height(), &result, cpu_frequency);

		cout << "Grayscale Image (Seq.) Results:" << endl << result << endl;

		if (displayImages) {
			grayImage.display("Grayscale Image (Seq.)");
		}
		if (saveAllImages) {
			if (debug) cout << "Saving image..." << endl;
			grayImage.save(string_format(output, "grayscale-seq").c_str());
			if (debug) cout << "Saved." << endl;
		}
		//Use pinned memory for GPU DMA
		cudaHostAlloc(&inputImagePinned, inputImage.size()* sizeof(unsigned char), cudaHostAllocMapped);
		cudaHostAlloc(&grayImagePinned, grayImage.size() * sizeof(unsigned char), cudaHostAllocMapped);

		memcpy(inputImagePinned, inputImage.data(), inputImage.size() * sizeof(unsigned char));

		rgb2grayCuda(inputImagePinned, grayImagePinned, inputImage.width(), inputImage.height(), &result, cpu_frequency);

		memcpy(grayImage.data(), grayImagePinned, grayImage.size() * sizeof(unsigned char));

		cudaFreeHost(inputImagePinned);
		cudaFreeHost(grayImagePinned);
		cout << "Grayscale Image (CUDA) Results:" << endl << result << endl;

		if (displayImages) {
			grayImage.display("Grayscale Image (CUDA)");
		}
		if (saveAllImages) {
			if (debug) cout << "Saving image..." << endl;
			grayImage.save(string_format(output, "grayscale-cuda").c_str());
			if (debug) cout << "Saved." << endl;
		}

		// Compute 1D histogram
		CImg< unsigned char > histogramImage = CImg< unsigned char >(barWidth * histogramSize, histogramSize, 1, 1);
		unsigned int *histogram_seq = new unsigned int[histogramSize];
		unsigned int *histogram_cuda = new unsigned int[histogramSize];

		histogram1D(grayImage.data(), histogramImage.data(), grayImage.width(), grayImage.height(), histogram_seq, histogramSize, barWidth, &result, cpu_frequency);
		cout << "Histogram (Seq.) Results:" << endl << result << endl;

		if (displayImages) {
			histogramImage.display("Histogram (Seq.)");
		}
		if (saveAllImages) {
			if (debug) cout << "Saving image..." << endl;
			histogramImage.save(string_format(output, "histogram-seq").c_str());
			if (debug) cout << "Saved." << endl;
		}

		//Use pinned memory for GPU DMA		
		cudaHostAlloc(&histogramPinned, histogramSize * sizeof(unsigned int), cudaHostAllocMapped);
		cudaHostAlloc(&grayImagePinned, grayImage.size() * sizeof(unsigned char), cudaHostAllocMapped);

		memcpy(grayImagePinned, grayImage.data(), grayImage.size() * sizeof(unsigned char));

		histogram1DCuda(grayImagePinned, histogramImage.data(), grayImage.width(), grayImage.height(), histogramPinned, histogramSize, barWidth, &result, cpu_frequency);

		memcpy(histogram_cuda, histogramPinned, histogramSize * sizeof(unsigned int));

		cudaFreeHost(histogramPinned);
		cudaFreeHost(grayImagePinned);
		cout << "Histogram (CUDA) Results:" << endl << result << endl;

		if (displayImages) {
			histogramImage.display("Histogram (CUDA)");
		}
		if (saveAllImages) {
			if (debug) cout << "Saving image..." << endl;
			histogramImage.save(string_format(output, "histogram-cuda").c_str());
			if (debug) cout << "Saved." << endl;
		}

		// Contrast enhancement
		contrast1D(grayImage.data(), grayImage.width(), grayImage.height(), histogram_seq, histogramSize, contrastThreshold, &result, cpu_frequency);
		cout << "Contrast Enhanced Image (Seq.) Results:" << endl << result << endl;

		if (displayImages) {
			grayImage.display("Contrast Enhanced Image (Seq.)");
		}
		if (saveAllImages) {
			if (debug) cout << "Saving image..." << endl;
			grayImage.save(string_format(output, "contrast-seq").c_str());
			if (debug) cout << "Saved." << endl;
		}

		//Use pinned memory for GPU DMA
		cudaHostAlloc(&grayImagePinned, grayImage.size() * sizeof(unsigned char), cudaHostAllocMapped);

		memcpy(grayImagePinned, grayImage.data(), grayImage.size() * sizeof(unsigned char));

		contrast1DCuda(grayImagePinned, grayImage.width(), grayImage.height(), histogram_cuda, histogramSize, contrastThreshold, &result, cpu_frequency);

		memcpy(grayImage.data(), grayImagePinned, grayImage.size() * sizeof(unsigned char));

		cudaFreeHost(grayImagePinned);

		cout << "Contrast Enhanced Image (CUDA) Results:" << endl << result << endl;

		if (displayImages) {
			grayImage.display("Contrast Enhanced Image (CUDA)");
		}
		if (saveAllImages) {
			if (debug) cout << "Saving image..." << endl;
			grayImage.save(string_format(output, "contrast-cuda").c_str());
			if (debug) cout << "Saved." << endl;
		}

		delete[] histogram_seq;
		delete[] histogram_cuda;

		// Triangular smooth (convolution)
		CImg< unsigned char > smoothImage = CImg< unsigned char >(grayImage.width(), grayImage.height(), 1, 1);

		triangularSmooth(grayImage.data(), smoothImage.data(), grayImage.width(), grayImage.height(), filter, &result, cpu_frequency);
		cout << "Smooth Image (Seq.) Results:" << endl << result << endl;

		if (displayImages) {
			smoothImage.display("Smooth Image (Seq.)");
		}

		if (saveAllImages) {
			if (debug) cout << "Saving image..." << endl;
			smoothImage.save(string_format(output, "smooth-seq").c_str());
			if (debug) cout << "Saved." << endl;
		}

		//Use pinned memory for GPU DMA
		cudaHostAlloc(&smoothImagePinned, smoothImage.size() * sizeof(unsigned char), cudaHostAllocMapped);
		cudaHostAlloc(&grayImagePinned, grayImage.size() * sizeof(unsigned char), cudaHostAllocMapped);
		cudaHostAlloc(&filterPinned, 25 * sizeof(float), cudaHostAllocMapped);

		memcpy(grayImagePinned, grayImage.data(), grayImage.size() * sizeof(unsigned char));
		memcpy(filterPinned, filter, 25 * sizeof(float));

		triangularSmoothCuda(grayImagePinned, smoothImagePinned, grayImage.width(), grayImage.height(), filterPinned, &result, cpu_frequency);

		memcpy(smoothImage.data(), smoothImagePinned, smoothImage.size() * sizeof(unsigned char));

		cudaFreeHost(smoothImagePinned);
		cudaFreeHost(grayImagePinned);
		cudaFreeHost(filterPinned);

		cout << "Smooth Image (CUDA) Results:" << endl << result << endl;

		if (displayImages) {
			smoothImage.display("Smooth Image (CUDA)");
		}

		if (saveAllImages) {
			if (debug) cout << "Saving image..." << endl;
			smoothImage.save(string_format(output, "smooth-cuda").c_str());
			if (debug) cout << "Saved." << endl;
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

