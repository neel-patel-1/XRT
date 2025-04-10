#ifndef IAA_OFFLOADS_H
#define IAA_OFFLOADS_H

#include "print_utils.h"
#include <cstdlib>
#include <cstring>

extern "C" {
  #include "iaa.h"
  #include "accel_test.h"
  #include "iaa_compress.h"
  #include "iaa_filter.h"
}

typedef struct completion_record ax_comp;
typedef struct hw_desc idxd_desc;
extern struct acctest_context *iaa;

void initialize_iaa_wq(int dev_id,
  int wq_id, int wq_type);
void free_iaa_wq();
void prepare_iaa_decompress_desc_with_preallocated_comp(
  struct hw_desc *hw, uint64_t src1, uint64_t dst1,
  uint64_t comp, uint64_t xfer_size );
void prepare_iaa_compress_desc_with_preallocated_comp(
  struct hw_desc *hw, uint64_t src1, uint64_t src2, uint64_t dst1,
  uint64_t comp, uint64_t xfer_size );
void prepare_iaa_filter_desc_with_preallocated_comp(
  struct hw_desc *hw, uint64_t src1, uint64_t dst1,
  uint64_t comp, uint64_t xfer_size );

#endif