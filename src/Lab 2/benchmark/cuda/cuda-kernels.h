#pragma once

#include "timing.h"
#include <iostream>
#include <cuda.h>
#include <iomanip>

using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

void rgb2grayCuda(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height, double cpu_frequency = 1);
void histogram1DCuda(unsigned char *grayImage, unsigned char *histogramImage, const int width, const int height, unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, const unsigned int BAR_WIDTH, double cpu_frequency = 1);
void contrast1DCuda(unsigned char *grayImage, const int width, const int height, unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, const unsigned int CONTRAST_THRESHOLD, double cpu_frequency = 1);
void triangularSmoothCuda(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height, const float *filter, double cpu_frequency = 1);
