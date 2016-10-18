#include "timing.h"

using namespace std;

//  Windows
#ifdef _WIN32

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
double get_frequency(bool debug) {
#if (defined __linux__ || defined __blrts__) && \
	(defined __i386__ || defined __x86_64__ || defined __ia64__ || defined __PPC__) && \
	(defined __GNUC__ || defined __INTEL_COMPILER || defined __PATHSCALE__ || defined __xlC__)
	std::ifstream infile("/proc/cpuinfo");
	char     buffer[256], *colon;

	while (infile.good()) {
		infile.getline(buffer, 256);

		if (strncmp("cpu MHz", buffer, 7) == 0 && (colon = strchr(buffer, ':')) != 0) {
			double freq = atof(colon + 2)*1e6;
			if(debug) cout << "Reported frequency (UNIX): " << (freq / 1e6) << " MHz" << endl;
			return freq;
		}
	}
	throw "Can't get frequency from proc/cpuinfo.";
#elif defined _WIN32
	LARGE_INTEGER frequency;
	if (::QueryPerformanceFrequency(&frequency) == FALSE)
		throw "Can't get frequency from QueryPerformanceFrequency.";

	//frequency is in kilo hertz
	if (debug) cout << "Reported frequency (Win32): " << frequency.QuadPart / 1e3 << " MHz" << endl;
	return (double)frequency.QuadPart * 1e3;
#endif
}
uint64_t now() {
	return rdtsc();
}
double diffToNanoseconds(uint64_t t1, uint64_t t2, double freq) {
	return (t2 - t1) / (freq) * 1e9;
}
#else
PerfClock::time_point now() {
	return PerfClock::now();
}
double diffToNanoseconds(PerfClock::time_point t1, PerfClock::time_point t2, double freq) {
	chrono::duration<double, nano> measurement = t2 - t1;
	return measurement.count();
}
#endif
