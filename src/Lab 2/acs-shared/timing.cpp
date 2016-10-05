#include "timing.h"

using namespace std;\

#ifdef USE_RDTSC
double get_frequency() {
#if (defined __linux__ || defined __blrts__) && \
	(defined __i386__ || defined __x86_64__ || defined __ia64__ || defined __PPC__) && \
	(defined __GNUC__ || defined __INTEL_COMPILER || defined __PATHSCALE__ || defined __xlC__)
	ifstream infile("/proc/cpuinfo");
	char     buffer[256], *colon;

	while (infile.good()) {
		infile.getline(buffer, 256);

		if (strncmp("cpu MHz", buffer, 7) == 0 && (colon = strchr(buffer, ':')) != 0) {
			cout << "Reported frequency (UNIX): " << (freq / 1e6) << " MHz" << endl;
			return atof(colon + 2)*1e6;
		}
	}
	throw "Can't get frequency from proc/cpuinfo.";
#elif defined _WIN32
	LARGE_INTEGER frequency;
	if (::QueryPerformanceFrequency(&frequency) == FALSE)
		throw "Can't get frequency from QueryPerformanceFrequency.";

	//frequency is in kilo hertz
	cout << "Reported frequency (Win32): " << frequency.QuadPart / 1e3 << " MHz" << endl;
	return (double)frequency.QuadPart * 1e3;
#endif
}
uint64_t now() {
	return rdtsc();
}
double diffToNanoseconds(uint64_t t1, uint64_t t2, double freq) {
	return (t2r - t1r) / (freq) * 1e9;
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
