#include <iostream>
#include <chrono>
#include <chrono_io.h>
#include <ratio>
#include <thread>
#include <vector>
#include <numeric>
#ifdef _WIN32
#include "Windows.h"
#endif
//for that hacky unix approach
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <cassert>
#include <cstdlib>
#include <cstring>

#include "Timer.hpp"

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

using namespace std;
using LOFAR::NSTimer;

template <class Clock>
void
display_precision() {
	typedef std::chrono::duration<double, std::nano> NS;
	NS ns = typename Clock::duration(1);
	std::cout << ns.count() << " ns" << endl;
}


static int sink = 0;

int main() {
	double freq = 0;
#if (defined __linux__ || defined __blrts__) && \
	(defined __i386__ || defined __x86_64__ || defined __ia64__ || defined __PPC__) && \
	(defined __GNUC__ || defined __INTEL_COMPILER || defined __PATHSCALE__ || defined __xlC__)
	ifstream infile("/proc/cpuinfo");
	char     buffer[256], *colon;

	while (infile.good()) {
		infile.getline(buffer, 256);
		cout << "Read from file: " << buffer << endl;
		if (strncmp("cpu MHz", buffer, 7) == 0 && (colon = strchr(buffer, ':')) != 0) {
			freq = atof(colon + 2)*1e6;
			cout << "Reported frequency (UNIX): " << (freq / 1e6) << " MHz" << endl;
			break;
		}		
	}
	if (freq==0) {
		cerr << "Can't get frequency from /proc/cpuinfo." << endl;
		throw "Can't get frequency from /proc/cpuinfo.";
	}

#elif defined _WIN32
	LARGE_INTEGER frequency;
	if (::QueryPerformanceFrequency(&frequency) == FALSE) {
		cerr << "Can't get frequency from QueryPerformanceFrequency." << endl;
		throw "Can't get frequency from QueryPerformanceFrequency.";
	}

	//frequency is in kilo hertz
	cout << "Reported frequency (Win32): " << frequency.QuadPart / 1e3 << " MHz" << endl;
	freq = (double)frequency.QuadPart * 1e3;
#endif

	cout << "Chrono min time: " << ((double)chrono::high_resolution_clock::period::num / chrono::high_resolution_clock::period::den)*1e9 << " ns" << endl;
	cout << "High resolution clock: ";
	display_precision<std::chrono::high_resolution_clock>();
	cout << "System clock: ";
	display_precision<std::chrono::system_clock>();

	cout << endl;
	//TODO test against NSTimer, because of the deep pipeline it seems also inaccurate.
	typedef chrono::high_resolution_clock PerfClock;
	
	int size = 250;
	int func_iter = 1000;
	int tests = 1000;
	unsigned long long int t1r, t2r;

	double avg_chrono = 0;
	double avg_rdtsc = 0;
	double avg_nstimer = 0;

	for (int test = 0;test < tests;test++) {
		NSTimer nsTimer = NSTimer("TestTimer", false, false);
		nsTimer.start();
		auto t1 = PerfClock::now();
		t1r = rdtsc();
		for (int i = 0;i < func_iter;i++) {
			std::vector<int> v(size, 42);
			sink = std::accumulate(v.begin(), v.end(), 0u);
		}
		t2r = rdtsc();
		auto t2 = PerfClock::now();
		nsTimer.stop();

		
		chrono::duration<double, nano> measurement = t2 - t1;

		double nstimer_ns = nsTimer.getElapsed()*1e9;
		avg_nstimer += nstimer_ns / tests;
		avg_chrono += measurement.count() / tests;
		double ns_rdtsc = (t2r - t1r) / (freq) * 1e9;
		avg_rdtsc += ns_rdtsc / tests;

		cout << "[" << test + 1 << "] NSTimer: f() took " << nstimer_ns << " ns for " << func_iter << " function iterations." << endl;
		cout << "[" << test + 1 << "] Chrono: f() took " << measurement.count() << " ns for " << func_iter << " function iterations." << endl;
		cout << "[" << test + 1 << "] RDTSC: f() took " << ns_rdtsc << " ns for " << func_iter << " function iterations." << endl;
		cout << "[" << test + 1 << "] and the result was " << sink << " for " << size << " vector size." << endl;
		cout << endl;

	}

	cout << endl;
	cout << "[AVG] NSTimer: f() took " << avg_nstimer << " ns." << endl;
	cout << "[AVG] Chrono: f() took " << avg_chrono << " ns." << endl;
	cout << "[AVG] RDTSC: f() took " << avg_rdtsc << " ns." << endl;
#ifdef _WIN32
	cout << "Press enter to continue." << endl;
	cin.get();
#endif
}