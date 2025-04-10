#include "dsa_offloads.h"

struct acctest_context *dsa;

void prepare_dsa_memcpy_desc_with_preallocated_comp(
  struct hw_desc *hw, uint64_t src,
  uint64_t dst, uint64_t comp, uint64_t xfer_size){

  memset(hw, 0, sizeof(struct hw_desc));
  hw->flags = IDXD_OP_FLAG_CRAV | IDXD_OP_FLAG_RCR | IDXD_OP_FLAG_BOF | IDXD_OP_FLAG_CC;
  hw->src_addr = src;
  hw->dst_addr = dst;
  hw->xfer_size = xfer_size;
  hw->opcode = DSA_OPCODE_MEMMOVE;

  memset((void *)comp, 0, sizeof(ax_comp));
  hw->completion_addr = comp;

}

void prepare_dsa_memfill_desc_with_preallocated_comp(
  struct hw_desc *hw, uint64_t pattern,
  uint64_t dst, uint64_t comp, uint64_t xfer_size){

  memset(hw, 0, sizeof(struct hw_desc));
  hw->flags = IDXD_OP_FLAG_CRAV | IDXD_OP_FLAG_RCR | IDXD_OP_FLAG_BOF;
  hw->src_addr = 0xdeadbeef;
  hw->dst_addr = dst;
  hw->xfer_size = xfer_size;
  hw->opcode = DSA_OPCODE_MEMFILL;

  memset((void *)comp, 0, sizeof(ax_comp));
  hw->completion_addr = comp;

}

void initialize_dsa_wq(int dev_id, int wq_id, int wq_type){
  int tflags = TEST_FLAGS_BOF;
  int rc;

  dsa = acctest_init(tflags);
  rc = acctest_alloc(dsa, wq_type, dev_id, wq_id);
  if(rc != ACCTEST_STATUS_OK){
    LOG_PRINT( LOG_ERR, "Error allocating work queue\n");
    exit(-1);
    return;
  }
}


void free_dsa_wq(){
  acctest_free(dsa);
}