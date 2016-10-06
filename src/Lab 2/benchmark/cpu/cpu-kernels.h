#pragma once
#include "timing.h"
#include <iostream>
#include <iomanip>
#include <cstring>
#include "../result_container.h"

using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

void rgb2gray(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height, ResultContainer *result, double cpu_frequency = 1);
void histogram1D(unsigned char *grayImage, unsigned char *histogramImage, const int width, const int height, unsigned int *histogram, const unsigned int histogramSize, const unsigned int barWidth, ResultContainer *result, double cpu_frequency = 1);
void contrast1D(unsigned char *grayImage, const int width, const int height, unsigned int *histogram, const unsigned int histogramSize, const unsigned int contrastThreshold, ResultContainer *result, double cpu_frequency = 1);
void triangularSmooth(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height, const float *filter, ResultContainer *result, double cpu_frequency = 1);
