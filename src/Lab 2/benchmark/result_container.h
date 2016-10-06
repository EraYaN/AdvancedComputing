#pragma once
#include "timing.h"
#include <iostream>
#include <iomanip>
#include <cmath>

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#ifndef BAR_LENGTH
#define BAR_LENGTH(part,whole)            (min(BAR_WIDTH,ceil(BAR_WIDTH * part / whole)) + 1)
#endif

#define COLUMN_WIDTH 22
#define BAR_WIDTH 40
#define REDUCTION_FACTOR 1000000
#define PRECISION 2
#define TIME_UNIT " ms"

// All times are in nanoseconds
struct ResultContainer {
	double preprocessing_time;
	double init_time;
	double kernel_time;
	double cleanup_time;
	double postprocessing_time;
	double total_time;

	friend std::ostream & operator<<(std::ostream &os, const ResultContainer& r) {
		return (os <<
			"\xC4\xC4 " << std::left << std::setw(COLUMN_WIDTH) << "Total time:" <<
			std::right << std::setw(COLUMN_WIDTH) << std::fixed << std::setprecision(PRECISION) << r.total_time / REDUCTION_FACTOR << TIME_UNIT << ';' <<
			std::right << std::setw(COLUMN_WIDTH) << std::fixed << std::setprecision(PRECISION) << r.total_time / r.total_time * 100 << " %" <<
			std::right << '\xC3' << std::setw(BAR_LENGTH(r.total_time,r.total_time)) << std::setfill('\xC4') << '\xB4' << std::setfill(' ') <<
			std::endl <<

			" \xDA " << std::left << std::setw(COLUMN_WIDTH) << "Pre-processing time:" <<
			std::right << std::setw(COLUMN_WIDTH) << std::fixed << std::setprecision(PRECISION) << r.preprocessing_time / REDUCTION_FACTOR << TIME_UNIT << ';' <<
			std::right << std::setw(COLUMN_WIDTH) << std::fixed << std::setprecision(PRECISION) << r.preprocessing_time / r.total_time * 100 << " %" <<
			std::right << '\xC3' << std::setw(BAR_LENGTH(r.preprocessing_time, r.total_time)) << std::setfill('\xC4') << '\xB4' << std::setfill(' ') <<
			std::endl <<

			" \xC3 " << std::left << std::setw(COLUMN_WIDTH) << "Init time:" <<
			std::right << std::setw(COLUMN_WIDTH) << std::fixed << std::setprecision(PRECISION) << r.init_time / REDUCTION_FACTOR << TIME_UNIT << ';' <<
			std::right << std::setw(COLUMN_WIDTH) << std::fixed << std::setprecision(PRECISION) << r.init_time / r.total_time * 100 << " %" <<
			std::right << '\xC3' << std::setw(BAR_LENGTH(r.init_time, r.total_time)) << std::setfill('\xC4') << '\xB4' << std::setfill(' ') <<
			std::endl <<

			"\xC4\xC5 " << std::left << std::setw(COLUMN_WIDTH) << "Kernel time:" <<
			std::right << std::setw(COLUMN_WIDTH) << std::fixed << std::setprecision(PRECISION) << r.kernel_time / REDUCTION_FACTOR << TIME_UNIT << ';' <<
			std::right << std::setw(COLUMN_WIDTH) << std::fixed << std::setprecision(PRECISION) << r.kernel_time / r.total_time * 100 << " %" <<
			std::right << '\xC3' << std::setw(BAR_LENGTH(r.kernel_time, r.total_time)) << std::setfill('\xC4') << '\xB4' << std::setfill(' ') <<
			std::endl <<

			" \xC3 " << std::left << std::setw(COLUMN_WIDTH) << "Cleanup time:" <<
			std::right << std::setw(COLUMN_WIDTH) << std::fixed << std::setprecision(PRECISION) << r.cleanup_time / REDUCTION_FACTOR << TIME_UNIT << ';' <<
			std::right << std::setw(COLUMN_WIDTH) << std::fixed << std::setprecision(PRECISION) << r.cleanup_time / r.total_time * 100 << " %" <<
			std::right << '\xC3' << std::setw(BAR_LENGTH(r.cleanup_time, r.total_time)) << std::setfill('\xC4') << '\xB4' << std::setfill(' ') <<
			std::endl <<

			" \xC0 " << std::left << std::setw(COLUMN_WIDTH) << "Post-processing time:" <<
			std::right << std::setw(COLUMN_WIDTH) << std::fixed << std::setprecision(PRECISION) << r.postprocessing_time / REDUCTION_FACTOR << TIME_UNIT << ';' <<
			std::right << std::setw(COLUMN_WIDTH) << std::fixed << std::setprecision(PRECISION) << r.postprocessing_time / r.total_time * 100 << " %" <<
			std::right << '\xC3' << std::setw(BAR_LENGTH(r.postprocessing_time, r.total_time)) << std::setfill('\xC4') << '\xB4' << std::setfill(' ')
			);
	}
	ResultContainer();

#ifdef USE_RDTSC
	ResultContainer(uint64_t t_preprocessing, uint64_t t_init, uint64_t t_kernel, uint64_t t_cleanup, uint64_t t_postprocessing, uint64_t t_end, double cpu_frequency = 1);

#else
	ResultContainer(PerfClock::time_point t_preprocessing, PerfClock::time_point t_init, PerfClock::time_point t_kernel, PerfClock::time_point t_cleanup, PerfClock::time_point t_postprocessing, PerfClock::time_point t_end, double cpu_frequency = 1);



#endif
};