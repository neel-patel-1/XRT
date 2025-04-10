#ifndef DSA_OFFLOADS_H
#define DSA_OFFLOADS_H

#include "print_utils.h"
#include <cstdlib>
#include <cstring>


extern "C" {
  #include "dsa.h"
  #include "accel_test.h"
}

typedef struct completion_record ax_comp;
typedef struct hw_desc idxd_desc;
extern struct acctest_context *dsa;

void initialize_dsa_wq(int dev_id, int wq_id, int wq_type);
void free_dsa_wq();
void prepare_dsa_memcpy_desc_with_preallocated_comp(
  struct hw_desc *hw, uint64_t src,
  uint64_t dst, uint64_t comp, uint64_t xfer_size);
void prepare_dsa_memfill_desc_with_preallocated_comp(
  struct hw_desc *hw, uint64_t src,
  uint64_t dst, uint64_t comp, uint64_t xfer_size);


#endif