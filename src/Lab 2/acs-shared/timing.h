#pragma once
#include <iostream>
#include <chrono>
#include <ratio>
#include <thread>
#include <vector>
#include <numeric>
#ifdef _WIN32
#include "Windows.h"
#endif

#include <stdint.h>

//  Windows
#ifdef _WIN32

#include <intrin.h>
uint64_t rdtsc() {
	return __rdtsc();
}

//  Linux/GCC
#else
uint64_t rdtsc() {
	unsigned int lo, hi;
	__asm__ __volatile__("rdtsc" : "=a" (lo), "=d" (hi));
	return ((uint64_t)hi << 32) | lo;
}

#endif

#ifdef USE_RDTSC
double get_frequency();
uint64_t now();
double diffToNanoseconds(uint64_t t1, uint64_t t2, double freq = 1);
#else
typedef std::chrono::high_resolution_clock PerfClock;
PerfClock::time_point now();
double diffToNanoseconds(PerfClock::time_point t1, PerfClock::time_point t2, double freq = 1);
#endif