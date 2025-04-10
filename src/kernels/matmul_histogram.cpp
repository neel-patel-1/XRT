#include "matmul_histogram.h"
#include "print_utils.h"
#include <cmath>



int swap;      // to indicate if we need to swap byte order of header information



/** matrix_mult()
 *  Blocked Matrix Multiply Function
 */
void matrix_mult(mm_data_t *data_in)
{
	assert(data_in);
	int i, j, k,a, b, c, end_i, end_j, end_k;

   LOG_PRINT(LOG_TOO_VERBOSE, "Matrix A\n");
	for(i = 0;i < (data_in->matrix_len)*data_in->matrix_len ; i++)
	{
		if(i%data_in->matrix_len == 0)
			LOG_PRINT(LOG_TOO_VERBOSE,"\n");
		LOG_PRINT(LOG_TOO_VERBOSE,"%d  ",data_in->matrix_A[i]);
	}
	LOG_PRINT(LOG_TOO_VERBOSE,"\n\n");

   LOG_PRINT(LOG_TOO_VERBOSE, "Matrix B\n");
	for(i = 0;i < (data_in->matrix_len)*data_in->matrix_len ; i++)
	{
		if(i%data_in->matrix_len == 0)
			LOG_PRINT(LOG_TOO_VERBOSE,"\n");
		LOG_PRINT(LOG_TOO_VERBOSE,"%d  ",data_in->matrix_B[i]);
	}
	LOG_PRINT(LOG_TOO_VERBOSE,"\n\n");

   for(i = 0; i < data_in->matrix_len; i += BLOCK_LEN)
   for(j = 0; j < data_in->matrix_len; j += BLOCK_LEN)
   for(k = 0; k < data_in->matrix_len; k += BLOCK_LEN)
   {
      end_i = i + BLOCK_LEN; end_j = j + BLOCK_LEN; end_k = k + BLOCK_LEN;
      for (a = i; a < end_i && a < data_in->matrix_len; a++)
      for (b = j; b < end_j && b < data_in->matrix_len; b++)
      for (c = k; c < end_k && c < data_in->matrix_len; c++)
      {
               data_in->matrix_out[(data_in->matrix_len)*a + b] +=
                  ( data_in->matrix_A[ (data_in->matrix_len)*a + c] *
                    data_in->matrix_B[ (data_in->matrix_len)*c + b]);
      }
   }

   LOG_PRINT(LOG_TOO_VERBOSE, "Resultant Matrix\n");
   for(i = 0; i < data_in->matrix_len; i++)
   {
      for(j = 0; j < data_in->matrix_len; j++)
      {
         LOG_PRINT(LOG_TOO_VERBOSE,"%d  ", data_in->matrix_out[(data_in->matrix_len)*i + j]);
      }

      LOG_PRINT(LOG_TOO_VERBOSE,"\n");
   }
}

void test_endianess() {
   unsigned int num = 0x12345678;
   char *low = (char *)(&(num));
   if (*low ==  0x78) {
      LOG_PRINT(LOG_DEBUG,"No need to swap\n");
      swap = 0;
   }
   else if (*low == 0x12) {
      LOG_PRINT(LOG_DEBUG,"Need to swap\n");
      swap = 1;
   }
   else {
      printf("Error: Invalid value found in memory\n");
      exit(1);
   }
}

/* swap_bytes
 *
 */
void swap_bytes(char *bytes, int num_bytes) {
   int i;
   char tmp;

   for (i = 0; i < num_bytes/2; i++) {
      LOG_PRINT(LOG_DEBUG,"Swapping %d and %d\n", bytes[i], bytes[num_bytes - i - 1]);
      tmp = bytes[i];
      bytes[i] = bytes[num_bytes - i - 1];
      bytes[num_bytes - i - 1] = tmp;
   }
}

void test_pre_proc(void *inp, void *output, int input_size, int *output_size){
  LOG_PRINT(LOG_DEBUG, "Test pre proc\n");
  int matlen, mat_size_bytes, ax_fill_size;

  matrix_mult((mm_data_t *)inp);

  matlen = ((mm_data_t *)inp)->matrix_len;
   mat_size_bytes = matlen * matlen * sizeof(int);
   ax_fill_size = mat_size_bytes / 2;

  *output_size = ax_fill_size;
}

void test_post_proc(void *inp, void *output, int input_size, int *output_size){
   LOG_PRINT(LOG_DEBUG, "Test post proc\n");

   int matlen;
   matlen = input_size;

   calc_mean((int *)inp, (int *)output, matlen, matlen);

   #ifndef PERF
   int *output_data = (int *)output;
   for(int i = 0; i < matlen; i++)
   {
      LOG_PRINT(LOG_TOO_VERBOSE,"%d  ", output_data[i]);
   }
   LOG_PRINT(LOG_TOO_VERBOSE,"\n");
   #endif


}

void calc_hist(void *inp, void *output, int input_size, int *output_size){
   int *outbuf = (int *)output;
   int *red;
   int *green;
   int *blue;
   int i;
   uint8_t *fdata = (uint8_t *)inp;

   red = (int *)(&(outbuf[0]));
   green = (int *)(&(outbuf[256]));
   blue = (int *)(&(outbuf[512]));

   uint8_t *data_pos = (uint8_t *)(&(fdata[IMG_DATA_OFFSET_POS]));
   uint8_t bitsperpixel = (uint8_t)(fdata[BITS_PER_PIXEL_POS]);

   memset(outbuf, 0, sizeof(int) * 256 * 3);
   for(i=*data_pos; i<input_size; i+=3){
      uint8_t *val = (uint8_t *)(&(fdata[i]));
      blue[*val]++;

      val = (uint8_t *)(&(fdata[i+1]));
      green[*val]++;

      val = (uint8_t *)(&(fdata[i+2]));
      red[*val]++;
   }

   #ifndef PERF
   for(i=0; i<256; i++){
      LOG_PRINT(LOG_TOO_VERBOSE, "Red[%d]: %d\n", i, red[i]);
      LOG_PRINT(LOG_TOO_VERBOSE, "Green[%d]: %d\n", i, green[i]);
      LOG_PRINT(LOG_TOO_VERBOSE, "Blue[%d]: %d\n", i, blue[i]);
   }
   #endif

   *output_size = (256 * 3 * sizeof(int));
}


/*
 * calc_mean()
 *  Compute the mean for each row
 */
void calc_mean(int *matrix, int *mean, int num_rows, int num_cols) {
   int i, j;
   int sum = 0;

   for (i = 0; i < num_rows; i++) {
      sum = 0;
      for (j = 0; j < num_cols; j++) {
         sum += matrix[i*num_cols + j];
      }
      mean[i] = sum / num_cols;
   }
}

/** calc_cov()
 *  Calculate the covariance
 */
void calc_cov(int **matrix, int *mean, int **cov, int num_rows, int num_cols) {
   int i, j, k;
   int sum;

   for (i = 0; i < num_rows; i++) {
      for (j = i; j < num_rows; j++) {
         sum = 0;
         for (k = 0; k < num_cols; k++) {
            sum = sum + ((matrix[i][k] - mean[i]) * (matrix[j][k] - mean[j]));
         }
         cov[i][j] = cov[j][i] = sum/(num_cols-1);
      }
   }
}