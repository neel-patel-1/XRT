#ifndef TEST_WRKLOAD
#define TEST_WRKLOAD
#include <stdint.h>
#include <stdio.h>
#include <strings.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <ctype.h>

extern int log_level;

#define IMG_DATA_OFFSET_POS 10
#define BITS_PER_PIXEL_POS 28

void test_pre_proc(void *inp, void *output, int input_size, int *output_size);
void test_post_proc(void *inp, void *output, int input_size, int *output_size);

/* matmul */
#define BLOCK_LEN 100
typedef struct {
   int *matrix_A;
   int *matrix_B;
   int *matrix_out;
   int matrix_len;
} mm_data_t;
void matrix_mult(mm_data_t *data_in);


/*histogram*/
extern int swap;
void test_endianess();
void swap_bytes(char *bytes, int num_bytes);
void calc_hist(void *inp, void *output, int input_size, int *output_size);

/* pca */
void calc_cov(int **matrix, int *mean, int **cov, int num_rows, int num_cols);
void calc_mean(int *matrix, int *mean, int num_rows, int num_cols);

#endif
