#include "sequential_writer.h"
#include "print_utils.h"
#include <immintrin.h>
#include <cstdint>

#define VECTOR_LOAD(x) _mm512_load_pd((void *)x);

void sequential_writer(char *l2_buf, uint64_t l2_buf_size){
  int l2_buf_idx = 0;
  volatile __m512d v;
  LOG_PRINT(LOG_DEBUG, "Filler buf size: %ld\n", l2_buf_size);
  if(l2_buf_size % 64 != 0){
    LOG_PRINT(LOG_ERR, "Buffer size not multiple of 64\n");
    return;
  }
  while(l2_buf_idx < l2_buf_size){
    v = VECTOR_LOAD( (l2_buf + l2_buf_idx) );
    l2_buf_idx += 64;
  }

  LOG_PRINT(LOG_DEBUG, "SeqWriterCompleted\n");

}