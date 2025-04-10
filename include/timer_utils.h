#ifndef TIMER_H
#define TIMER_H

#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>
#include <x86intrin.h>

#define UPPER_HALF_OF_REGISTER 32

static inline uint64_t sampleCoderdtsc(void)
{
    volatile unsigned long a, d;

    asm volatile("lfence;rdtsc" : "=a"(a), "=d"(d));
    asm volatile("lfence;");
    return (((uint64_t)a) | (((uint64_t)d) << UPPER_HALF_OF_REGISTER));
}

static inline void gen_diff_array(uint64_t *dst_array, uint64_t* array1,  uint64_t* array2, int size)
{
  for(int i=0; i<size; i++){ dst_array[i] = array2[i] - array1[i]; }
}

#define do_sum_array(accum,array,iter) accum = 0; \
 for (int i=1; i<iter; i++){ accum+=array[i]; } \
 accum /= iter

#define do_avg(sum, itr) (sum/itr)

#define avg_samples_from_arrays(yield_to_submit, avg, after, before, num_samples) \
  gen_diff_array(yield_to_submit, before, after, num_samples); \
  do_sum_array(avg, yield_to_submit, num_samples); \
  do_avg(avg, num_samples);


#define time_code_region(per_run_setup, code_to_measrure, per_run_cleanup, iterations) \
  uint64_t start, end, avg; \
  uint64_t start_times[iterations], end_times[iterations], times[iterations]; \
  for(int i=0; i<iterations; i++){ \
    per_run_setup; \
    start = sampleCoderdtsc(); \
    code_to_measrure; \
    end = sampleCoderdtsc(); \
    start_times[i] = start; \
    end_times[i] = end; \
    per_run_cleanup; \
  } \
  avg_samples_from_arrays(times, avg, end_times, start_times, iterations); \
  PRINT("AverageCycles: %ld\n", avg);

#define rerun_code_region(iterations, code) \
  for(int i=0; i<iterations; i++){ \
    code; \
  }

#endif