#pragma once

#include "timing.h"
#include <iostream>
#include <cuda.h>
#include <iomanip>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../result_container.h"
#include "../checkCudaCall.h"

using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

void rgb2grayCuda(unsigned char *inputImage, unsigned char *grayImage, unsigned char *dev_grayImage, const int width, const int height, ResultContainer *result, double cpu_frequency = 1);
void histogram1DCuda(unsigned char *grayImage, unsigned char *dev_grayImage, unsigned char *histogramImage, const int width, const int height, unsigned int *histogram, const unsigned int histogramSize, const unsigned int barWidth, ResultContainer *result, double cpu_frequency = 1);
void contrast1DCuda(unsigned char *grayImage, unsigned char *dev_grayImage, const int width, const int height, unsigned int *histogram, const unsigned int histogramSize, const unsigned int contrastThreshold, ResultContainer *result, double cpu_frequency = 1);
void triangularSmoothCuda(unsigned char *grayImage, unsigned char *dev_grayImage, unsigned char *smoothImage, const int width, const int height, float *filter, ResultContainer *result, double cpu_frequency = 1);
