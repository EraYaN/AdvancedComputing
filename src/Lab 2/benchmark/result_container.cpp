#include "result_container.h"

// All times are in nanoseconds

ResultContainer::ResultContainer() : preprocessing_time(0), init_time(0), kernel_time(0), cleanup_time(0), postprocessing_time(0), total_time(0) {
}

#ifdef USE_RDTSC
ResultContainer::ResultContainer(uint64_t t_preprocessing, uint64_t t_init, uint64_t t_kernel, uint64_t t_cleanup, uint64_t t_postprocessing, uint64_t t_end, double cpu_frequency) :
	preprocessing_time(diffToNanoseconds(t_preprocessing, t_init, cpu_frequency)), init_time(diffToNanoseconds(t_init, t_kernel, cpu_frequency)),
	kernel_time(diffToNanoseconds(t_kernel, t_cleanup, cpu_frequency)), cleanup_time(diffToNanoseconds(t_cleanup, t_postprocessing, cpu_frequency)),
	postprocessing_time(diffToNanoseconds(t_postprocessing, t_end, cpu_frequency)), total_time(diffToNanoseconds(t_preprocessing, t_end, cpu_frequency)) {
}
#else
ResultContainer::ResultContainer(PerfClock::time_point t_preprocessing, PerfClock::time_point t_init, PerfClock::time_point t_kernel, PerfClock::time_point t_cleanup, PerfClock::time_point t_postprocessing, PerfClock::time_point t_end, double cpu_frequency) :
	preprocessing_time(diffToNanoseconds(t_preprocessing, t_init, cpu_frequency)), init_time(diffToNanoseconds(t_init, t_kernel, cpu_frequency)),
	kernel_time(diffToNanoseconds(t_kernel, t_cleanup, cpu_frequency)), cleanup_time(diffToNanoseconds(t_cleanup, t_postprocessing, cpu_frequency)),
	postprocessing_time(diffToNanoseconds(t_postprocessing, t_end, cpu_frequency)), total_time(diffToNanoseconds(t_preprocessing, t_end, cpu_frequency)) {
}
#endif

