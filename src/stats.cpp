#include "stats.h"
#include "print_utils.h"
#include <cmath>

uint64_t avg_from_array(uint64_t *arr, int size){
  uint64_t sum = 0;
  for(int i=0; i<size; i++)
    sum += arr[i];
  return sum / size;

}

double stddev_from_array(uint64_t *arr, int size){
  uint64_t avg = avg_from_array(arr, size);
  uint64_t sum = 0;
  for(int i=0; i<size; i++)
    sum += (arr[i] - avg) * (arr[i] - avg);
  double stdev = sqrt(sum / size);
  return stdev;
}

uint64_t median_from_array(uint64_t *arr, int size){
  uint64_t *sorted = (uint64_t *)malloc(sizeof(uint64_t) * size);
  memcpy(sorted, arr, sizeof(uint64_t) * size);
  std::sort(sorted, sorted + size);
  uint64_t median = sorted[size / 2];
  free(sorted);
  return median;
}

void print_mean_median_stdev(uint64_t *arr, int size, const char *name){
  uint64_t mean = avg_from_array(arr, size);
  uint64_t median = median_from_array(arr, size);
  double stdev = stddev_from_array(arr, size);
  PRINT( "%s Mean: %lu Median: %lu Stddev: %f\n", name, mean, median, stdev);
}

void mean_median_stdev_rps(uint64_t *cycle_arr, int size, int total_requests, const char *name){
  uint64_t mean = avg_from_array(cycle_arr, size);
  uint64_t median = median_from_array(cycle_arr, size);
  double stdev = stddev_from_array(cycle_arr, size);

  double rpsmean = (double)total_requests / (mean / 2100000000.0);
  double rpsmedian = (double)total_requests / (median / 2100000000.0);
  double rpsstddev = (double)total_requests / (stdev / 2100000000.0);

  PRINT( "%s Mean: %f Median: %f Stddev: %f\n", name, rpsmean, rpsmedian, rpsstddev);
}