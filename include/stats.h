#ifndef STATS_H
#define STATS_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>

uint64_t avg_from_array(uint64_t *arr, int size);
double stddev_from_array(uint64_t *arr, int size);
uint64_t median_from_array(uint64_t *arr, int size);
void print_mean_median_stdev(uint64_t *arr, int size, const char *name);
void mean_median_stdev_rps(uint64_t *cycle_arr, int size, int total_requests, const char *name);
#endif