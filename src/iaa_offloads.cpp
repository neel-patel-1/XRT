#include "iaa_offloads.h"
#include "print_utils.h"
#include <cstdio>

struct acctest_context *iaa;

void initialize_iaa_wq(int dev_id, int wq_id, int wq_type){
  int tflags = TEST_FLAGS_BOF;
  int rc;

  iaa = acctest_init(tflags);
  rc = acctest_alloc(iaa, wq_type, dev_id, wq_id);
  if(rc != ACCTEST_STATUS_OK){
    LOG_PRINT( LOG_ERR, "Error allocating work queue\n");
    exit(-1);
    return;
  }
}

void free_iaa_wq(){
  acctest_free(iaa);
}

void prepare_iaa_compress_desc_with_preallocated_comp(
  struct hw_desc *hw, uint64_t src1, uint64_t src2, uint64_t dst1,
  uint64_t comp, uint64_t xfer_size )
{
  memset(hw, 0, sizeof(struct hw_desc));
  hw->flags = 0x5000eUL;
  hw->opcode = 0x43;
  hw->src_addr = src1;
  hw->dst_addr = dst1;
  hw->xfer_size = xfer_size;

  memset((void *)comp, 0, sizeof(ax_comp));
  hw->completion_addr = comp;
  hw->iax_compr_flags = 14;
  hw->iax_src2_addr = src2;
  hw->iax_src2_xfer_size = IAA_COMPRESS_AECS_SIZE;
  hw->iax_max_dst_size = IAA_COMPRESS_MAX_DEST_SIZE;
}

void prepare_iaa_decompress_desc_with_preallocated_comp(
  struct hw_desc *hw, uint64_t src1, uint64_t dst1,
  uint64_t comp, uint64_t xfer_size )
{
  memset(hw, 0, sizeof(struct hw_desc));
  hw->flags = 14 | IDXD_OP_FLAG_CC;
  hw->opcode = IAX_OPCODE_DECOMPRESS;
  hw->src_addr = src1;
  hw->dst_addr = dst1;
  hw->xfer_size = xfer_size;

  memset((void *)comp, 0, sizeof(ax_comp));
  hw->completion_addr = comp;
  hw->iax_decompr_flags = 31;
  hw->iax_src2_addr = 0x0;
  hw->iax_src2_xfer_size = 0;
  hw->iax_max_dst_size = IAA_COMPRESS_MAX_DEST_SIZE;
}

void prepare_iaa_filter_desc_with_preallocated_comp(
  struct hw_desc *hw, uint64_t src1, uint64_t dst1,
  uint64_t comp, uint64_t xfer_size ){

  uint32_t low_val = 0;
  uint32_t num_inputs = xfer_size;
  uint32_t high_val = num_inputs/2;

  uint8_t *aecs =
    (uint8_t *)aligned_alloc(IAA_FILTER_AECS_SIZE, IAA_FILTER_AECS_SIZE);

  if(!aecs){
    LOG_PRINT( LOG_ERR, "Error allocating memory for aecs\n");
    exit(-1);
  }

  struct iaa_filter_aecs_t iaa_filter_aecs =
  {
    .rsvd = 0,
    .rsvd2 = 0,
    .rsvd3 = 0,
    .rsvd4 = 0,
    .rsvd5 = 0,
    .rsvd6 = 0
  };

  /* prepare aecs */
  memset(aecs, 0, IAA_FILTER_AECS_SIZE);
  iaa_filter_aecs.low_filter_param = low_val;
  iaa_filter_aecs.high_filter_param = high_val;
  memcpy(aecs, (void *)&iaa_filter_aecs, IAA_FILTER_AECS_SIZE);

  /* prepare hw */
  memset(hw, 0, sizeof(struct hw_desc));
  hw->flags |= (IDXD_OP_FLAG_CRAV | IDXD_OP_FLAG_RCR);
  hw->flags |= IDXD_OP_FLAG_BOF;
  hw->flags |= IDXD_OP_FLAG_RD_SRC2_AECS;
  hw->opcode = IAX_OPCODE_EXTRACT;
  hw->src_addr = src1;
  hw->dst_addr = dst1;
  hw->xfer_size = xfer_size;

  /* prepare hw filter params */
  hw->iax_num_inputs = num_inputs;
  hw->iax_filter_flags = 28;
  hw->iax_src2_addr = (uint64_t)aecs;
  hw->iax_src2_xfer_size = IAA_FILTER_AECS_SIZE;
  hw->iax_max_dst_size = IAA_FILTER_MAX_DEST_SIZE;

  /* comp */
  memset((void *)comp, 0, sizeof(ax_comp));
  hw->completion_addr = comp;

  #ifndef PERF
  for(int i = 0; i < sizeof(hw_desc)/sizeof(uint64_t); i++)
  {
    LOG_PRINT(LOG_DEBUG, "0x%016lx\n", ((uint64_t *)hw)[i]);
  }
  #endif

}