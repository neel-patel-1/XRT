#ifndef GATHER_H
#define GATHER_H

void gather_access(void *inp, void *output, int input_size, int *output_size);
void gather_using_indir_array(float *input,
  float *output,  int *indir_arr, int num_accesses);
void scatter_update_inplace(void *inp, void *output, int input_size, int *output_size);
void scatter_update_inplace_using_indir_array(void *inp, int *indir_arr, int num_acc);

#endif