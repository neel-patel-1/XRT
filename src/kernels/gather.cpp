#include "gather.h"
#include "print_utils.h"
#include <cstdint>

/*
indir array size: sizeof(int) * num_accesses
input size: (max value in indir_arr) + 1
output size: sizeof(float) * num_accesses
*/
void gather_using_indir_array(float *input,
  float *output,  int *indir_arr, int num_acc)
{
  for(int i=0; i < num_acc; i++){
    output[i] = input[indir_arr[i]];
    LOG_PRINT(LOG_TOO_VERBOSE, "input[%d] -> output[%d]: %f\n",indir_arr[i], i, output[i]);
  }
}


void scatter_update_inplace_using_indir_array(void *inp, int *indir_arr, int num_acc)
{
  uint8_t *input = (uint8_t *)inp;
  for(int i=0; i < num_acc; i++){
    input[indir_arr[i]]++;
  }
}