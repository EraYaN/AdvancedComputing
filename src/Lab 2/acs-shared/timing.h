#pragma once
#include <iostream>
#ifdef USE_RDTSC
#include <sstream>
#include <fstream>
#include <cstring>
#else
#include <chrono>
#include <ratio>
#endif
#ifdef _WIN32
#include "Windows.h"
#include <intrin.h>
#endif

#include <stdint.h>
#include <stdlib.h>

//  Windows#ifdef _WIN32


#ifdef USE_RDTSC
double get_frequency(bool debug);
uint64_t now();
double diffToNanoseconds(uint64_t t1, uint64_t t2, double freq = 1);
#else
typedef std::chrono::high_resolution_clock PerfClock;
PerfClock::time_point now();
double diffToNanoseconds(PerfClock::time_point t1, PerfClock::time_point t2, double freq = 1);
#endif