#include "numa_mem.h"
#include "payload_gen.h"
#include "print_utils.h"
#include "stats.h"
#include <x86intrin.h>
#include <immintrin.h>
#include <sched.h>
#include <sys/sysinfo.h>
#include "src/protobuf/generated/src/protobuf/router.pb.h"
#include "ch3_hash.h"
#include "gather.h"
#include "pointer_chase.h"
#include "iaa_offloads.h"
#include "dsa_offloads.h"
#include "matmul_histogram.h"
#include "payload_gen.h"
#include "gpcore_compress.h"
#include "lzdatagen.h"
#include "decrypt.h"
#include "dotproduct.h"
#include "test.h"
#include "gather.h"
extern "C" {
  #include "fcontext.h"
  #include "idxd.h"
  #include "accel_test.h"
}

#define L1_SIZE 48 * 1024
#define L3_SIZE 75 * 1024 * 1024
#define GB_SIZE 1024 * 1024 * 1024
constexpr uint64_t buf_llc_size = 78643200;
constexpr uint64_t buf_1gb_size = 1073741824;
constexpr uint64_t buf_4gb_size = 4294967296;

#define VECTOR_LOAD(x) _mm512_load_pd((void *)x);

#define READ_ONCE(x) (*(volatile typeof(x) *)&(x))
#define MAX_SER_OVERHEAD_BYTES 64

typedef struct completion_record idxd_comp;
typedef struct hw_desc idxd_desc;
typedef struct serial_accessor_args {
  idxd_desc *desc;
  idxd_comp *comp;
  void *src;
  void *dst;
  void **dl_buf;
  uint64_t xfer_size;
  uint64_t uncomp_size; /* for decomp user */
  uint64_t dl_size;
  int p_off;
  uint64_t *ts0;
  uint64_t *ts1;
  uint64_t *ts2;
  uint64_t *ts3;
  uint64_t *ts4;
} dl_args_t;
typedef struct deser_decomp_hash_args {
  idxd_desc *desc;
  idxd_comp *comp;
  char *s_buf;
  char *d_buf;
  int s_sz;
  uint32_t hash;
  int d_sz;

  int p_off;
  uint64_t *ts0;
  uint64_t *ts1;
  uint64_t *ts2;
  uint64_t *ts3;
  uint64_t *ts4;
} ddh_args_t;
typedef struct decomp_gather_args {
  idxd_desc *desc;
  idxd_comp *comp;
  char *s_buf;
  int *g_buf;
  float *d_buf;
  float *o_buf;
  int s_sz;
  int d_sz;
  int num_accesses;

  int p_off;
  uint64_t *ts0;
  uint64_t *ts1;
  uint64_t *ts2;
  uint64_t *ts3;
  uint64_t *ts4;
} dg_args_t;
typedef struct decrypt_memcpy_dp_args {
  idxd_desc *desc;
  idxd_comp *comp;
  char *enc_buf;
  char *dec_buf;
  char *dst_buf;
  float score;
  int sz;

  int p_off;
  uint64_t *ts0;
  uint64_t *ts1;
  uint64_t *ts2;
  uint64_t *ts3;
  uint64_t *ts4;
} dmdp_args_t;

typedef struct matmul_memfill_pca_args {
  idxd_desc *desc;
  idxd_comp *comp;
  mm_data_t *mm_data;
  int *mean_vec;
  int mat_size_bytes;

  int p_off;
  uint64_t *ts0;
  uint64_t *ts1;
  uint64_t *ts2;
  uint64_t *ts3;
  uint64_t *ts4;
} mmpc_args_t;

typedef struct update_filter_histogram_args {
  idxd_desc *desc;
  idxd_comp *comp;
  int *scat_buf;
  int num_acc;
  float *upd_buf;
  uint8_t *extracted;
  uint8_t *hist;
  uint8_t *aecs;
  int low_val;
  int high_val;

  int mat_size_bytes;

  int p_off;
  uint64_t *ts0;
  uint64_t *ts1;
  uint64_t *ts2;
  uint64_t *ts3;
  uint64_t *ts4;
} ufh_args_t;

enum cache_state {
  L2_DIRTY,
  L2_CLEAN,
  LLC,
  DRAM,
  L2_DIRTY_DEMOTE,
  L2_CLEAN_DEMOTE,
  LLC_DEMOTE,
  DRAM_DEMOTE,
  CORE_NT
};

typedef struct dsa_offload_args{
  enum dsa_opcode opcode;
  char * src;
  char * dst;
  enum cache_state c_state;
  idxd_desc *desc;
  idxd_comp *comp;
  int xfer_size;
  int p_off;

  uint64_t *ts0;
  uint64_t *ts1;
  uint64_t *ts2;
  uint64_t *ts3;
  uint64_t *ts4;
} dsa_args_t;

typedef struct iaa_offload_args{
  enum iax_opcode opcode;
  char * src;
  char * dst;
  enum cache_state c_state;
  idxd_desc *desc;
  idxd_comp *comp;
  int xfer_size;
  int p_off;

  uint64_t *ts0;
  uint64_t *ts1;
  uint64_t *ts2;
  uint64_t *ts3;
  uint64_t *ts4;
} iaa_args_t;

typedef struct antagonist_args {
  void *input;
  uint64_t input_size;
  bool done;
} a_args_t;

__thread fcontext_transfer_t scheduler_xfer;
__thread volatile uint8_t *preempt_signal;
__thread request_return_args_t ret_args;
IppsAES_GCMState *pState = NULL;

void
flush_range(void *start, size_t len)
{
  char *ptr = (char *)start;
  char *end = ptr + len;

  for (; ptr < end; ptr += 64) {
    _mm_clflush(ptr);
  }
}

int
cpu_pin(uint32_t cpu)
{
	cpu_set_t *cpuset;
	size_t cpusetsize;

	cpusetsize = CPU_ALLOC_SIZE(get_nprocs());
	cpuset = CPU_ALLOC(get_nprocs());
	CPU_ZERO_S(cpusetsize, cpuset);
	CPU_SET_S(cpu, cpusetsize, cpuset);

	pthread_setaffinity_np(pthread_self(), cpusetsize, cpuset);

	CPU_FREE(cpuset);

	return 0;
}

static __always_inline uint64_t
rdtsc(void)
{
	uint64_t tsc;
	unsigned int dummy;

	/*
	 * https://www.felixcloutier.com/x86/rdtscp
	 * The RDTSCP instruction is not a serializing instruction, but it
	 * does wait until all previous instructions have executed and all
	 * previous loads are globally visible
	 *
	 * If software requires RDTSCP to be executed prior to execution of
	 * any subsequent instruction (including any memory accesses), it can
	 * execute LFENCE immediately after RDTSCP
	 */
  _mm_mfence();
	tsc = __rdtscp(&dummy);
  _mm_mfence();
	// __builtin_ia32_lfence();

	return tsc;
}

extern "C" void do_yield()
{
  LOG_PRINT(LOG_DEBUG, "Yielding\n");
  ret_args.status = REQUEST_PREEMPTED;
  scheduler_xfer = fcontext_swap(scheduler_xfer.prev_context, NULL);


}

void indirect_array_populate(int *indirect_array, int num_accesses, int min_val, int max_val){
  // int num_floats_in_array = array_size_bytes / sizeof(float);
  // int max_val = num_floats_in_array - 1;
  // int min_val = 0;
  for(int i=0; i<num_accesses; i++){
    int idx = (rand() % (max_val - min_val + 1)) + min_val;
    indirect_array[i] = idx;
  }
}

void create_random_chain_in_place(int size, void **memory){ /* only touches each cacheline*/
  uint64_t len = size / 64;
  uint64_t  *indices =
    (uint64_t *)malloc(sizeof(uint64_t) * len);
  for (int i = 0; i < len; i++) {
    indices[i] = i;
  }
  random_permutation(indices, len); /* have a random permutation of cache lines to pick*/

  /* the memaddr is 8 bytes -- only read each cache line once */
  for (int i = 1; i < len; ++i) {
    memory[indices[i-1] * 8] = (void *) &memory[indices[i] * 8];
  }
  memory[indices[len - 1] * 8] = (void *) &memory[indices[0] * 8 ];
}

void create_random_chain_starting_at(int size, void **memory, void **st_addr){ /* only touches each cacheline*/
  uint64_t len = size / 64;
  uint64_t  *indices = (uint64_t *)malloc(sizeof(uint64_t) * len);
  for (int i = 0; i < len; i++) {
    indices[i] = i;
  }
  random_permutation(indices, len); /* have a random permutation of cache lines to pick*/

  for (int i = 1; i < len; ++i) {
    memory[indices[i-1] * 8] = (void *) &st_addr[indices[i] * 8];
  }
  memory[indices[len - 1] * 8] = (void *) &st_addr[indices[0] * 8 ];
}

void enc_buf(Ipp8u *out, Ipp8u *in, int size){
  int ippAES_GCM_ctx_size;
  IppStatus status;

  status = ippsAES_GCMGetSize(&ippAES_GCM_ctx_size);
  if(status != ippStsNoErr){
    LOG_PRINT(LOG_ERR, "Failed to get AES GCM size\n");
  }

  if(pState != NULL){
    free(pState);
  }
  pState = (IppsAES_GCMState *)malloc(ippAES_GCM_ctx_size);
  int keysize = 16;
  int ivsize = 12;
  int aadSize = 16;
  int taglen = 16;
  Ipp8u *pKey = (Ipp8u *)"0123456789abcdef";
  Ipp8u *pIV = (Ipp8u *)"0123456789ab";
  Ipp8u *pAAD = (Ipp8u *)malloc(aadSize);
  Ipp8u *pTag = (Ipp8u *)malloc(taglen);

  LOG_PRINT(LOG_TOO_VERBOSE, "Plaintext: %x\n", *(uint32_t *)in);

  status = ippsAES_GCMInit(pKey, keysize, pState, ippAES_GCM_ctx_size);
  if(status != ippStsNoErr){
    LOG_PRINT(LOG_ERR, "Failed to init AES GCM\n");
  }

  status = ippsAES_GCMStart(pIV, ivsize, pAAD, aadSize, pState);
  if(status != ippStsNoErr){
    LOG_PRINT(LOG_ERR, "Failed to start AES GCM\n");
  }

  status = ippsAES_GCMEncrypt(in, out, size, pState);
  if(status != ippStsNoErr){
    LOG_PRINT(LOG_ERR, "Failed to encrypt AES GCM\n");
  }

  status = ippsAES_GCMGetTag(pTag, taglen, pState);
  if(status != ippStsNoErr){
    LOG_PRINT(LOG_ERR, "Failed to get tag AES GCM\n");
  }

  LOG_PRINT(LOG_TOO_VERBOSE, "Ciphertext: %x\n", *(uint32_t *)out);
}

static inline void decrypt_feature(void *cipher_inp, void *plain_out, int input_size){
  Ipp8u *pKey = (Ipp8u *)"0123456789abcdef";
  Ipp8u *pIV = (Ipp8u *)"0123456789ab";
  int keysize = 16;
  int ivsize = 12;
  int aadSize = 16;
  Ipp8u aad[aadSize];
  IppStatus status;

  status = ippsAES_GCMReset(pState);
  if(status != ippStsNoErr){
    LOG_PRINT(LOG_ERR, "Failed to reset AES GCM\n");
  }
  status = ippsAES_GCMStart(pIV, ivsize, aad, aadSize, pState);
  if(status != ippStsNoErr){
    LOG_PRINT(LOG_ERR, "Failed to start AES GCM\n");
  }
  status = ippsAES_GCMDecrypt((Ipp8u *)cipher_inp, (Ipp8u *)plain_out, input_size, pState);
  if(status != ippStsNoErr){
    LOG_PRINT(LOG_ERR, "Failed to decrypt AES GCM: %d\n", status);
  }

  LOG_PRINT(LOG_TOO_VERBOSE, "Decrypted: %x\n", *(uint32_t *)plain_out);
}

void sequential_writer_uninterruptible(char *l2_buf, uint64_t l2_buf_size){
  int l2_buf_idx = 0;
  volatile __m512d v;
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

void filler_cache_evict_uninterruptible(fcontext_transfer_t arg){
  a_args_t *args = (a_args_t *)arg.data;
  char *inp = (char *)args->input;
  uint64_t inp_size = args->input_size;

  scheduler_xfer = arg;

  sequential_writer_uninterruptible(inp, inp_size);

  args->done = true;

  fcontext_swap(arg.prev_context, NULL);
}


void filler_cache_evict_interruptible(fcontext_transfer_t arg){
  a_args_t *args = (a_args_t *)arg.data;
  char *inp = (char *)args->input;
  uint64_t inp_size = args->input_size;

  scheduler_xfer = arg;

  sequential_writer(inp, inp_size);

  args->done = true;

  fcontext_swap(arg.prev_context, NULL);
}

static inline void stream_into_cache(void *buf, int size){
  for(int i = 0; i < size; i+=64){
    READ_ONCE(((char *)buf)[i]);
  }
}

void serial_accessor_dsa_memcpy_blocking(fcontext_transfer_t arg){
  struct serial_accessor_args *args =
    (struct serial_accessor_args *)arg.data;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  void *src = args->src;
  void *dst = args->dst;
  uint64_t xfer_size = args->xfer_size;
  int p_off = args->p_off;

  void **dl_buf = args->dl_buf;
  uint64_t dl_size = args->dl_size;
  int num_cl = dl_size / 64;

  uint64_t ts0, ts1, ts2, ts3, ts4;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  stream_into_cache(dl_buf, dl_size);

#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_dsa_memcpy_desc_with_preallocated_comp(
    desc, (uint64_t)src, (uint64_t)dst,
    (uint64_t)comp, xfer_size);

  while(enqcmd((void *)((char *)(dsa->wq_reg) + p_off), desc) ){
    /* retry submit */
  }

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  while(comp->status == IAX_COMP_NONE){  }

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  chase_pointers(dl_buf, num_cl);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void serial_accessor_iaa_decompress_blocking(fcontext_transfer_t arg){
  struct serial_accessor_args *args =
    (struct serial_accessor_args *)arg.data;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  void *src = args->src;
  void *dst = args->dst;
  uint64_t xfer_size = args->xfer_size;
  int p_off = args->p_off;

  void **dl_buf = args->dl_buf;
  uint64_t dl_size = args->dl_size;
  int num_cl = dl_size / 64;

  uint64_t ts0, ts1, ts2, ts3, ts4;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  stream_into_cache(dl_buf, dl_size);

#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_iaa_decompress_desc_with_preallocated_comp(
    desc, (uint64_t)src, (uint64_t)dst,
    (uint64_t)comp, xfer_size);

  while(enqcmd((void *)((char *)(iaa->wq_reg) + p_off), desc) ){
    /* retry submit */
  }

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  while(comp->status == IAX_COMP_NONE){  }
  if(comp->status != IAX_COMP_SUCCESS){
    LOG_PRINT(LOG_ERR, "Decompression failed: %d\n", comp->status);
    return;
  }

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  chase_pointers(dl_buf, num_cl);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void serial_accessor_axdep_iaa_decompress_blocking(fcontext_transfer_t arg){
  struct serial_accessor_args *args =
    (struct serial_accessor_args *)arg.data;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  void *src = args->src;
  void *dst = args->dst;
  uint64_t xfer_size = args->xfer_size;
  int p_off = args->p_off;

  void **dl_buf = args->dl_buf;
  uint64_t dl_size = args->dl_size;
  uint64_t uncomp_size = args->uncomp_size;
  int num_cl = dl_size / 64;

  uint64_t offset = uncomp_size - dl_size;
  void *chase_buf;

  uint64_t ts0, ts1, ts2, ts3, ts4;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  stream_into_cache(dl_buf, dl_size);

#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_iaa_decompress_desc_with_preallocated_comp(
    desc, (uint64_t)src, (uint64_t)dst,
    (uint64_t)comp, xfer_size);

  while(enqcmd((void *)((char *)(iaa->wq_reg) + p_off), desc) ){
    /* retry submit */
  }

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  while(comp->status == IAX_COMP_NONE){  }
  if(comp->status != IAX_COMP_SUCCESS){
    LOG_PRINT(LOG_ERR, "Decompression failed: %d\n", comp->status);
    return;
  }

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  chase_buf = (void *) ((char *)dst + offset);
  chase_pointers((void **)chase_buf, num_cl);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void serial_accessor_axdep_iaa_decompress_baseline(fcontext_transfer_t arg){
  struct serial_accessor_args *args =
    (struct serial_accessor_args *)arg.data;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  void *src = args->src;
  void *dst = args->dst;
  uint64_t xfer_size = args->xfer_size;
  int p_off = args->p_off;
  uLong avail_out = IAA_DECOMPRESS_MAX_DEST_SIZE;

  void **dl_buf = args->dl_buf;
  uint64_t dl_size = args->dl_size;
  uint64_t uncomp_size = args->uncomp_size;
  int num_cl = dl_size / 64;

  uint64_t offset = uncomp_size - dl_size;
  void *chase_buf;

  uint64_t ts0, ts1, ts2, ts3, ts4;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  stream_into_cache(dl_buf, dl_size);

#ifdef EXETIME
  ts1 = rdtsc();
#endif
#ifdef EXETIME
  ts2 = rdtsc();
#endif

  gpcore_do_decompress(dst, src, xfer_size, &avail_out);

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  chase_buf = (void *) ((char *)dst + offset);
  chase_pointers((void **)chase_buf, num_cl);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void serial_accessor_axdep_iaa_decompress_yielding(fcontext_transfer_t arg){
  struct serial_accessor_args *args =
    (struct serial_accessor_args *)arg.data;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  void *src = args->src;
  void *dst = args->dst;
  uint64_t xfer_size = args->xfer_size;
  int p_off = args->p_off;

  void **dl_buf = args->dl_buf;
  uint64_t dl_size = args->dl_size;
  uint64_t uncomp_size = args->uncomp_size;
  int num_cl = dl_size / 64;

  uint64_t offset = uncomp_size - dl_size;
  void *chase_buf;

  uint64_t ts0, ts1, ts2, ts3, ts4;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  stream_into_cache(dl_buf, dl_size);

#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_iaa_decompress_desc_with_preallocated_comp(
    desc, (uint64_t)src, (uint64_t)dst,
    (uint64_t)comp, xfer_size);

  while(enqcmd((void *)((char *)(iaa->wq_reg) + p_off), desc) ){
    /* retry submit */
  }

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  fcontext_swap(arg.prev_context, NULL);

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  chase_buf = (void *) ((char *)dst + offset);
  chase_pointers((void **)chase_buf, num_cl);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void serial_accessor_iaa_decompress_yielding(fcontext_transfer_t arg){
  struct serial_accessor_args *args =
    (struct serial_accessor_args *)arg.data;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  void *src = args->src;
  void *dst = args->dst;
  uint64_t xfer_size = args->xfer_size;
  int p_off = args->p_off;

  void **dl_buf = args->dl_buf;
  uint64_t dl_size = args->dl_size;
  int num_cl = dl_size / 64;

  uint64_t ts0, ts1, ts2, ts3, ts4;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  stream_into_cache(dl_buf, dl_size);

#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_iaa_decompress_desc_with_preallocated_comp(
    desc, (uint64_t)src, (uint64_t)dst,
    (uint64_t)comp, xfer_size);

  while(enqcmd((void *)((char *)(iaa->wq_reg) + p_off), desc) ){
    /* retry submit */
  }

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  fcontext_swap(arg.prev_context, NULL);

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  chase_pointers(dl_buf, num_cl);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void serial_accessor_dsa_memcpy_baseline(fcontext_transfer_t arg){
  struct serial_accessor_args *args =
    (struct serial_accessor_args *)arg.data;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  void *src = args->src;
  void *dst = args->dst;
  uint64_t xfer_size = args->xfer_size;
  int p_off = args->p_off;

  void **dl_buf = args->dl_buf;
  uint64_t dl_size = args->dl_size;
  int num_cl = dl_size / 64;

  uint64_t ts0, ts1, ts2, ts3, ts4;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  stream_into_cache(dl_buf, dl_size);

#ifdef EXETIME
  ts1 = rdtsc();
#endif

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  memcpy(dst, src, xfer_size);

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  chase_pointers(dl_buf, num_cl);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void serial_accessor_dsa_memcpy_yielding(fcontext_transfer_t arg){
  struct serial_accessor_args *args =
    (struct serial_accessor_args *)arg.data;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  void *src = args->src;
  void *dst = args->dst;
  uint64_t xfer_size = args->xfer_size;
  int p_off = args->p_off;

  void **dl_buf = args->dl_buf;
  uint64_t dl_size = args->dl_size;
  int num_cl = dl_size / 64;

  uint64_t ts0, ts1, ts2, ts3, ts4;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  stream_into_cache(dl_buf, dl_size);

#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_dsa_memcpy_desc_with_preallocated_comp(
    desc, (uint64_t)src, (uint64_t)dst,
    (uint64_t)comp, xfer_size);

  while(enqcmd((void *)((char *)(dsa->wq_reg) + p_off), desc) ){
    /* retry submit */
  }

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  fcontext_swap(arg.prev_context, NULL);

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  chase_pointers(dl_buf, num_cl);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void serial_accessor_axdep_dsa_memcpy_yielding(fcontext_transfer_t arg){
  struct serial_accessor_args *args =
    (struct serial_accessor_args *)arg.data;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  void *src = args->src;
  void *dst = args->dst;
  uint64_t xfer_size = args->xfer_size;
  int p_off = args->p_off;

  void **dl_buf = args->dl_buf;
  uint64_t dl_size = args->dl_size;
  int num_cl = dl_size / 64;

  uint64_t offset = xfer_size - dl_size;
  void *chase_buf;

  uint64_t ts0, ts1, ts2, ts3, ts4;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  stream_into_cache(dl_buf, L1_SIZE);

#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_dsa_memcpy_desc_with_preallocated_comp(
    desc, (uint64_t)src, (uint64_t)dst,
    (uint64_t)comp, xfer_size);

  while(enqcmd((void *)((char *)(dsa->wq_reg) + p_off), desc) ){
    /* retry submit */
  }

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  while(comp->status == IAX_COMP_NONE){  }
  fcontext_swap(arg.prev_context, NULL);

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  chase_buf = (void *) ((char *)dst + offset);
  chase_pointers((void **)chase_buf, num_cl);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void serial_accessor_axdep_dsa_memcpy_baseline(fcontext_transfer_t arg){
  struct serial_accessor_args *args =
    (struct serial_accessor_args *)arg.data;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  void *src = args->src;
  void *dst = args->dst;
  uint64_t xfer_size = args->xfer_size;
  int p_off = args->p_off;

  void **dl_buf = args->dl_buf;
  uint64_t dl_size = args->dl_size;
  int num_cl = dl_size / 64;

  uint64_t offset = xfer_size - dl_size;
  void *chase_buf;

  uint64_t ts0, ts1, ts2, ts3, ts4;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  stream_into_cache(dl_buf, L1_SIZE);

#ifdef EXETIME
  ts1 = rdtsc();
#endif
#ifdef EXETIME
  ts2 = rdtsc();
#endif

  memcpy(dst, src, xfer_size);

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  chase_buf = (void *) ((char *)dst + offset);
  chase_pointers((void **)chase_buf, num_cl);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void serial_accessor_axdep_dsa_memcpy_blocking(fcontext_transfer_t arg){
  struct serial_accessor_args *args =
    (struct serial_accessor_args *)arg.data;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  void *src = args->src;
  void *dst = args->dst;
  uint64_t xfer_size = args->xfer_size;
  int p_off = args->p_off;

  void **dl_buf = args->dl_buf;
  uint64_t dl_size = args->dl_size;
  int num_cl = dl_size / 64;

  uint64_t offset = xfer_size - dl_size;
  void *chase_buf;

  uint64_t ts0, ts1, ts2, ts3, ts4;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  stream_into_cache(dl_buf, L1_SIZE);

#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_dsa_memcpy_desc_with_preallocated_comp(
    desc, (uint64_t)src, (uint64_t)dst,
    (uint64_t)comp, xfer_size);

  while(enqcmd((void *)((char *)(dsa->wq_reg) + p_off), desc) ){
    /* retry submit */
  }

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  while(comp->status == IAX_COMP_NONE){  }
  if(comp->status != IAX_COMP_SUCCESS){
    LOG_PRINT(LOG_ERR, "Decompression failed: %d\n", comp->status);
    return;
  }

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  chase_buf = (void *) ((char *)dst + offset);
  chase_pointers((void **)chase_buf, num_cl);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}


void gen_comp_buf(int des_psize, int avail_out, void *compbuf, int *outsize, double target_ratio){
  double len_exp = 3.0; // linear distribution of lengths
  double lit_exp = 3.0; // linear distribution of literals

  char *valbuf;
  uint64_t msgsize;
  int compsize = avail_out;
  int rc;

  int req_uncompsize = des_psize * target_ratio;
  valbuf = (char *)malloc(req_uncompsize);

  lzdg_generate_reuse_buffers((void *)valbuf, req_uncompsize, target_ratio, len_exp, lit_exp);

  /* get compress bound*/
  rc = gpcore_do_compress(compbuf, (void *)valbuf, req_uncompsize, &compsize);
  if(rc != Z_OK){
    LOG_PRINT(LOG_ERR, "Compression failed\n");
    return;
  }
  LOG_PRINT(LOG_DEBUG, "CompSize: %d\n", compsize);

  *outsize = compsize;
}


void gen_ser_buf(int avail_out, char *p_val, char *dst, int *msg_size, int insize){
  router::RouterRequest req;

  req.set_key("/region/cluster/foo:key|#|etc"); // key is 32B string, value gets bigger up to 2MB
  req.set_operation(0);

  std::string valstring(p_val, insize);

  req.set_value(valstring);

  *msg_size = req.ByteSizeLong();
  if(*msg_size > avail_out){
    LOG_PRINT(LOG_ERR, "Not enough space to serialize\n");
    return;
  }
  req.SerializeToArray(dst, *msg_size);
}

/*
  create a compressed buffer of desired compressed size des_psize with
  ratio 3.0 and place in serialized buffer
*/
void gen_ser_comp_payload(
  void *out, /* the output buffer */
  int des_psize, /*how big we want the payload */
  int max_comp_size, /* how big to alloccate for the comp buf */
  int avail_ser_out, /* how much space is available in the output */
  int *outsize, /* how big the serialized payload is */
  double target_ratio){

  void *comp_buf = (void *)malloc(max_comp_size);
  int comp_size;
  int ser_size;

  gen_comp_buf(des_psize, max_comp_size, comp_buf, &comp_size, target_ratio);

  gen_ser_buf(avail_ser_out, (char *)comp_buf, (char *)out, &ser_size, comp_size);
  LOG_PRINT(LOG_DEBUG, "Serialized Payload Size: %d\n", ser_size);

  *outsize = ser_size;

  free(comp_buf);
}

void deser_decomp_hash_yielding(fcontext_transfer_t arg){
  struct deser_decomp_hash_args *args =
    (struct deser_decomp_hash_args *)arg.data;

  router::RouterRequest req;
  int p_off = args->p_off;
  int s_sz = args->s_sz;
  char *s_buf = (char *)args->s_buf;
  char *d_buf = (char *)args->d_buf;
  int d_sz = args->d_sz;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;

  scheduler_xfer = arg;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  req.ParseFromArray(s_buf, s_sz);
  LOG_PRINT(LOG_VERBOSE, "Deserialized Payload Size: %ld\n", req.value().size());

#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_iaa_decompress_desc_with_preallocated_comp(
    desc, (uint64_t)(&(req.value())[0]), (uint64_t)d_buf,
    (uint64_t)comp, req.value().size());

  while(enqcmd((void *)((char *)(iaa->wq_reg) + p_off), desc) ){
    /* retry submit */
  }

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  #ifdef SOJOURN
  ret_args.status = REQUEST_YIELDED;
  #endif
  fcontext_swap(arg.prev_context, NULL);

  LOG_PRINT(LOG_TOO_VERBOSE, "DecompData[0:5]: %c%c%c%c%c\n", d_buf[0], d_buf[1], d_buf[2], d_buf[3], d_buf[4]);
  LOG_PRINT(LOG_TOO_VERBOSE, "DecompData[Last5]: %c%c%c%c%c\n", d_buf[d_sz-5], d_buf[d_sz-4], d_buf[d_sz-3], d_buf[d_sz-2], d_buf[d_sz-1]);

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  args->hash = furc_hash(d_buf, d_sz, 16);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif


  ret_args.status = REQUEST_COMPLETED;
  fcontext_swap(arg.prev_context, NULL);
}

void deser_decomp_hash_baseline(fcontext_transfer_t arg){
  struct deser_decomp_hash_args *args =
    (struct deser_decomp_hash_args *)arg.data;

  router::RouterRequest req;
  int p_off = args->p_off;
  int s_sz = args->s_sz;
  char *s_buf = (char *)args->s_buf;
  char *d_buf = (char *)args->d_buf;
  int d_sz = args->d_sz;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  uLong d_out_spc = IAA_DECOMPRESS_MAX_DEST_SIZE;

  scheduler_xfer = arg;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  req.ParseFromArray(s_buf, s_sz);
  LOG_PRINT(LOG_VERBOSE, "Deserialized Payload Size: %ld\n", req.value().size());

#ifdef EXETIME
  ts1 = rdtsc();
#endif

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  gpcore_do_decompress(d_buf,
    (char *)(&(req.value())[0]),
    req.value().size(), &d_out_spc);

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  args->hash = furc_hash(d_buf, d_sz, 16);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}


void deser_decomp_hash_blocking(fcontext_transfer_t arg){
  struct deser_decomp_hash_args *args =
    (struct deser_decomp_hash_args *)arg.data;

  router::RouterRequest req;
  int p_off = args->p_off;
  int s_sz = args->s_sz;
  char *s_buf = (char *)args->s_buf;
  char *d_buf = (char *)args->d_buf;
  int d_sz = args->d_sz;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  req.ParseFromArray(s_buf, s_sz);
  LOG_PRINT(LOG_VERBOSE, "Deserialized Payload Size: %ld\n", req.value().size());

#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_iaa_decompress_desc_with_preallocated_comp(
    desc, (uint64_t)(&(req.value())[0]), (uint64_t)d_buf,
    (uint64_t)comp, req.value().size());

  while(enqcmd((void *)((char *)(iaa->wq_reg) + p_off), desc) ){
    /* retry submit */
  }

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  while(comp->status == IAX_COMP_NONE){  }
  if(comp->status != IAX_COMP_SUCCESS){
    LOG_PRINT(LOG_ERR, "Decompression failed: %d\n", comp->status);
    return;
  }
  LOG_PRINT(LOG_TOO_VERBOSE, "DecompData[0:5]: %c%c%c%c%c\n", d_buf[0], d_buf[1], d_buf[2], d_buf[3], d_buf[4]);
  LOG_PRINT(LOG_TOO_VERBOSE, "DecompData[Last5]: %c%c%c%c%c\n", d_buf[d_sz-5], d_buf[d_sz-4], d_buf[d_sz-3], d_buf[d_sz-2], d_buf[d_sz-1]);

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  args->hash = furc_hash(d_buf, d_sz, 16);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void decomp_gather_baseline(fcontext_transfer_t arg){
struct decomp_gather_args *args =
    (struct decomp_gather_args *)arg.data;

  int p_off = args->p_off;
  int s_sz = args->s_sz;
  char *s_buf = args->s_buf;
  float *d_buf = args->d_buf;
  int *g_buf = args->g_buf;
  float *o_buf = args->o_buf;
  int num_accesses = args->num_accesses;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  int rc;
  uLong d_sz = IAA_DECOMPRESS_MAX_DEST_SIZE;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

#ifdef EXETIME
  ts1 = rdtsc();
#endif

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  rc = gpcore_do_decompress(d_buf, s_buf, s_sz, &d_sz);

  LOG_PRINT(LOG_TOO_VERBOSE, "DecompData[0:5]: %f%f%f%f%f\n", d_buf[0], d_buf[1], d_buf[2], d_buf[3], d_buf[4]);
  LOG_PRINT(LOG_TOO_VERBOSE, "DecompData[Last5]: %f%f%f%f%f\n", d_buf[d_sz-5], d_buf[d_sz-4], d_buf[d_sz-3], d_buf[d_sz-2], d_buf[d_sz-1]);

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  LOG_PRINT(LOG_DEBUG, "NumAccesses: %d\n", num_accesses);
  chase_pointers((void **)d_buf, d_sz/64);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void decomp_gather_blocking(fcontext_transfer_t arg){
  struct decomp_gather_args *args =
    (struct decomp_gather_args *)arg.data;

  int p_off = args->p_off;
  int s_sz = args->s_sz;
  char *s_buf = args->s_buf;
  float *d_buf = args->d_buf;
  int *g_buf = args->g_buf;
  float *o_buf = args->o_buf;
  int num_accesses = args->num_accesses;
  int d_sz = args->d_sz;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

#ifdef EXETIME
  ts1 = rdtsc();
#endif

  LOG_PRINT(LOG_DEBUG, "Decompressing %d bytes\n", s_sz);
  prepare_iaa_decompress_desc_with_preallocated_comp(
    desc, (uint64_t)(s_buf), (uint64_t)d_buf,
    (uint64_t)comp, s_sz);

  while(enqcmd((void *)((char *)(iaa->wq_reg) + p_off), desc) ){
    /* retry submit */
  }

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  while(comp->status == IAX_COMP_NONE){  }
  if(comp->status != IAX_COMP_SUCCESS){
    LOG_PRINT(LOG_ERR, "Decompression failed: %d\n", comp->status);
    return;
  }
  LOG_PRINT(LOG_TOO_VERBOSE, "DecompData[0:5]: %f%f%f%f%f\n", d_buf[0], d_buf[1], d_buf[2], d_buf[3], d_buf[4]);
  LOG_PRINT(LOG_TOO_VERBOSE, "DecompData[Last5]: %f%f%f%f%f\n", d_buf[d_sz-5], d_buf[d_sz-4], d_buf[d_sz-3], d_buf[d_sz-2], d_buf[d_sz-1]);

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  LOG_PRINT(LOG_DEBUG, "NumAccesses: %d\n", num_accesses);
  chase_pointers((void **)d_buf, d_sz/64);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void decomp_gather_yielding(fcontext_transfer_t arg){
  struct decomp_gather_args *args =
    (struct decomp_gather_args *)arg.data;

  int p_off = args->p_off;
  int s_sz = args->s_sz;
  char *s_buf = args->s_buf;
  float *d_buf = args->d_buf;
  int *g_buf = args->g_buf;
  float *o_buf = args->o_buf;
  int num_accesses = args->num_accesses;
  int d_sz = args->d_sz;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;

  scheduler_xfer = arg;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_iaa_decompress_desc_with_preallocated_comp(
    desc, (uint64_t)(s_buf), (uint64_t)d_buf,
    (uint64_t)comp, s_sz);

  while(enqcmd((void *)((char *)(iaa->wq_reg) + p_off), desc) ){
    return;
  }

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  fcontext_swap(arg.prev_context, NULL);
  LOG_PRINT(LOG_TOO_VERBOSE, "DecompData[0:5]: %f%f%f%f%f\n", d_buf[0], d_buf[1], d_buf[2], d_buf[3], d_buf[4]);
  LOG_PRINT(LOG_TOO_VERBOSE, "DecompData[Last5]: %f%f%f%f%f\n", d_buf[d_sz-5], d_buf[d_sz-4], d_buf[d_sz-3], d_buf[d_sz-2], d_buf[d_sz-1]);

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  chase_pointers((void **)d_buf, d_sz/64);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

static inline void write_to_buf(char *buf, int size){
  for(int i = 0; i < size; i+=64){
    ((volatile char *)(buf))[i] = 'a';
  }
}

static inline void demote_buf(char *buf, int size){
  _mm_mfence();
  for(int i = 0; i < size; i+=64){
    _cldemote((void *)&buf[i]);
  }
    _mm_mfence();

}

void dsa_offload(fcontext_transfer_t arg){
  dsa_args_t *args =
    (dsa_args_t *)arg.data;

  void *src = args->src;
  void *dst = args->dst;
  uint64_t xfer_size = args->xfer_size;
  int p_off = args->p_off;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  enum cache_state c_state = args->c_state;

  if(c_state != CORE_NT){
    stream_into_cache(src, xfer_size);
  } else {
    flush_range(src, xfer_size);
  }

  if(c_state == L2_DIRTY){
    write_to_buf((char *)src, xfer_size);
  }
  if(c_state == CORE_NT){
    int c = 0x41;
    __m128i i = _mm_set_epi8(c, c, c, c,
                            c, c, c, c,
                            c, c, c, c,
                            c, c, c, c);
    char *p = (char *)src;
    for(int j=0; j<xfer_size; j+=64){
      _mm_stream_si128((__m128i *)&p[j + 0], i);
      _mm_stream_si128((__m128i *)&p[j + 16], i);
      _mm_stream_si128((__m128i *)&p[j + 32], i);
      _mm_stream_si128((__m128i *)&p[j + 48], i);
    }
  }
  if(c_state == LLC || c_state == LLC_DEMOTE){
    demote_buf((char *)src, xfer_size);
  }
  if(c_state == DRAM){
    flush_range(src, xfer_size);
  }

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  switch(c_state){
    case L2_DIRTY:
      break;
    case L2_CLEAN:
      break;
    case LLC:
      break;
    case DRAM:
      break;
    case L2_DIRTY_DEMOTE:
      demote_buf((char *)src, xfer_size);
      break;
    case L2_CLEAN_DEMOTE:
      demote_buf((char *)src, xfer_size);
      break;
    case LLC_DEMOTE:
      demote_buf((char *)src, xfer_size);
      break;
    case DRAM_DEMOTE:
      demote_buf((char *)src, xfer_size);
      break;
    case CORE_NT:
      break;
    default:
      LOG_PRINT(LOG_ERR, "Invalid cache state\n");
      return;
  }

#ifdef EXETIME
  ts1 = rdtsc();
#endif

  switch(args->opcode){
    case DSA_OPCODE_MEMMOVE:
      prepare_dsa_memcpy_desc_with_preallocated_comp(
        desc, (uint64_t)src, (uint64_t)dst,
        (uint64_t)comp, xfer_size);
      break;
    case DSA_OPCODE_MEMFILL:
      prepare_dsa_memfill_desc_with_preallocated_comp(
        desc, 0xdeadbeef, (uint64_t)src,
        (uint64_t)comp, xfer_size);
      break;
    default:
      LOG_PRINT(LOG_ERR, "Invalid DSA opcode\n");
      return;
  }

  while(enqcmd((void *)((char *)(dsa->wq_reg) + p_off), desc) ){
    /* retry submit */
  }

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  while(comp->status == IAX_COMP_NONE){  }
  if(comp->status != IAX_COMP_SUCCESS){
    LOG_PRINT(LOG_ERR, "DSA operation failed: %d\n", comp->status);
    return;
  }

#ifdef EXETIME
  ts3 = rdtsc();
#endif

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void iaa_offload(fcontext_transfer_t arg){
  iaa_args_t *args =
    (iaa_args_t *)arg.data;

  void *src = args->src;
  void *dst = args->dst;
  uint64_t xfer_size = args->xfer_size;
  int p_off = args->p_off;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  enum cache_state c_state = args->c_state;

  /* decompress and recompress into the input in case to preserve and dirty the data */
  void *dirty_state_temp_decomp_buf;
  uLong d_out_spc = IAA_DECOMPRESS_MAX_DEST_SIZE;
  int avail_comp_out = xfer_size;

  stream_into_cache(src, xfer_size);

  if(c_state == L2_DIRTY || c_state == LLC){
    dirty_state_temp_decomp_buf = (void *)malloc(IAA_DECOMPRESS_MAX_DEST_SIZE);
      gpcore_do_decompress(
        dirty_state_temp_decomp_buf,
        (char *)src, xfer_size, &d_out_spc
      );
      gpcore_do_compress((char *)src, dirty_state_temp_decomp_buf,
        d_out_spc, &avail_comp_out);
      if(avail_comp_out != xfer_size){
        LOG_PRINT(LOG_ERR, "Compression failed\n");
        return;
      }
  }

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  switch(c_state){
    case L2_DIRTY:
      break;
    case L2_CLEAN:
      break;
    case LLC:
      demote_buf((char *)src, xfer_size);
      break;
    case DRAM:
      flush_range(src, xfer_size);
      break;
    default:
      LOG_PRINT(LOG_ERR, "Invalid cache state\n");
      return;
  }

#ifdef EXETIME
  ts1 = rdtsc();
#endif

  switch(args->opcode){
    case IAX_OPCODE_DECOMPRESS:
      prepare_iaa_decompress_desc_with_preallocated_comp(
        desc, (uint64_t)src, (uint64_t)dst,
        (uint64_t)comp, xfer_size);
      break;
    case IAX_OPCODE_EXTRACT:
      prepare_iaa_filter_desc_with_preallocated_comp(
        desc,(uint64_t)src, (uint64_t)dst, (uint64_t)comp, xfer_size);
      break;
    default:
      LOG_PRINT(LOG_ERR, "Invalid IAA opcode\n");
      return;
  }
  while(enqcmd((void *)((char *)(iaa->wq_reg) + p_off), desc) ){
    /* retry submit */
  }

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  while(comp->status == IAX_COMP_NONE){  }
  if(comp->status != IAX_COMP_SUCCESS){
    LOG_PRINT(LOG_ERR, "IAA operation failed: %d\n", comp->status);
    return;
  }

#ifdef EXETIME
  ts3 = rdtsc();
#endif

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void memcpy_gather_baseline(fcontext_transfer_t arg){
  struct decomp_gather_args *args =
    (struct decomp_gather_args *)arg.data;

  int p_off = args->p_off;
  int s_sz = args->s_sz;
  char *s_buf = args->s_buf;
  float *d_buf = args->d_buf;
  int *g_buf = args->g_buf;
  float *o_buf = args->o_buf;
  int num_accesses = args->num_accesses;
  int d_sz = args->d_sz;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

#ifdef EXETIME
  ts1 = rdtsc();
#endif

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  memcpy(d_buf, s_buf, s_sz);

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  // gather_using_indir_array(d_buf, o_buf, g_buf, num_accesses);
  chase_pointers((void **)d_buf, d_sz/64);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void memcpy_gather_yielding(fcontext_transfer_t arg){
  struct decomp_gather_args *args =
    (struct decomp_gather_args *)arg.data;

  int p_off = args->p_off;
  int s_sz = args->s_sz;
  char *s_buf = args->s_buf;
  float *d_buf = args->d_buf;
  int *g_buf = args->g_buf;
  float *o_buf = args->o_buf;
  int num_accesses = args->num_accesses;
  int d_sz = args->d_sz;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;


#ifdef EXETIME
  ts0 = rdtsc();
#endif

#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_dsa_memcpy_desc_with_preallocated_comp(
    desc, (uint64_t)(s_buf), (uint64_t)d_buf,
    (uint64_t)comp, s_sz);
  while(enqcmd((void *)((char *)(dsa->wq_reg) + p_off), desc) ){
    /* retry submit */
    exit(1);
  }

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  fcontext_swap(arg.prev_context, NULL);

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  chase_pointers((void **)d_buf, d_sz/64);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void memcpy_gather_blocking(fcontext_transfer_t arg){
  struct decomp_gather_args *args =
    (struct decomp_gather_args *)arg.data;

  int p_off = args->p_off;
  int s_sz = args->s_sz;
  char *s_buf = args->s_buf;
  float *d_buf = args->d_buf;
  int *g_buf = args->g_buf;
  float *o_buf = args->o_buf;
  int num_accesses = args->num_accesses;
  int d_sz = args->d_sz;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_dsa_memcpy_desc_with_preallocated_comp(
    desc, (uint64_t)(s_buf), (uint64_t)d_buf,
    (uint64_t)comp, s_sz);
  while(enqcmd((void *)((char *)(dsa->wq_reg) + p_off), desc) ){
    /* retry submit */
  }

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  while(comp->status == IAX_COMP_NONE){  }
  if(comp->status != IAX_COMP_SUCCESS){
    LOG_PRINT(LOG_ERR, "Memcpy failed: %d\n", comp->status);
    return;
  }

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  chase_pointers((void **)d_buf, d_sz/64);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void decrypt_memcpy_dotproduct_blocking(fcontext_transfer_t arg){
  dmdp_args_t *args =
    (dmdp_args_t *)arg.data;

  char *enc_buf = args->enc_buf;
  char *dec_buf = args->dec_buf;
  char *dst_buf = args->dst_buf;
  int sz = args->sz;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  int out_sz;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  decrypt_feature(enc_buf, dec_buf, sz);

#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_dsa_memcpy_desc_with_preallocated_comp(
    desc, (uint64_t)dec_buf, (uint64_t)dst_buf,
    (uint64_t)comp, sz);
  while(enqcmd((void *)((char *)(dsa->wq_reg) + args->p_off), desc) ){}

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  while(comp->status == IAX_COMP_NONE){  }

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  dotproduct(dst_buf, &(args->score), sz, &sz);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);

}

void decrypt_memcpy_dotproduct_yielding(fcontext_transfer_t arg){
  dmdp_args_t *args =
    (dmdp_args_t *)arg.data;

  char *enc_buf = args->enc_buf;
  char *dec_buf = args->dec_buf;
  char *dst_buf = args->dst_buf;
  int sz = args->sz;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  int out_sz;

  scheduler_xfer = arg;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  decrypt_feature(enc_buf, dec_buf, sz);

#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_dsa_memcpy_desc_with_preallocated_comp(
    desc, (uint64_t)dec_buf, (uint64_t)dst_buf,
    (uint64_t)comp, sz);
  while(enqcmd((void *)((char *)(dsa->wq_reg) + args->p_off), desc) ){}

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  fcontext_swap(arg.prev_context, NULL);

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  ret_args.status = REQUEST_YIELDED;
  dotproduct(dst_buf, &(args->score), sz, &sz);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  ret_args.status = REQUEST_COMPLETED;
  fcontext_swap(arg.prev_context, NULL);
}

void decrypt_memcpy_dotproduct_baseline(fcontext_transfer_t arg){
  dmdp_args_t *args =
    (dmdp_args_t *)arg.data;

  char *enc_buf = args->enc_buf;
  char *dec_buf = args->dec_buf;
  char *dst_buf = args->dst_buf;
  int sz = args->sz;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  int out_sz;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  decrypt_feature(enc_buf, dec_buf, sz);

#ifdef EXETIME
  ts1 = rdtsc();
#endif

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  memcpy(dst_buf, dec_buf, sz);

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  dotproduct(dst_buf, &(args->score), sz, &sz);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void matmul_memfill_pca_baseline(fcontext_transfer_t arg){
  mmpc_args_t *args =
    (mmpc_args_t *)arg.data;

  mm_data_t *mm_data = args->mm_data;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  int *mean_vec = args->mean_vec;
  int mat_size_bytes = args->mat_size_bytes;


#ifdef EXETIME
  ts0 = rdtsc();
#endif

  matrix_mult(mm_data);

#ifdef EXETIME
  ts1 = rdtsc();
#endif

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  memset_pattern((void *)mm_data->matrix_out, 0xdeadbeef, mat_size_bytes/2);

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  calc_mean(
    mm_data->matrix_out,
    mean_vec,
    mm_data->matrix_len,
    mm_data->matrix_len
  );

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void matmul_memfill_pca_blocking(fcontext_transfer_t arg){
  mmpc_args_t *args =
    (mmpc_args_t *)arg.data;

  mm_data_t *mm_data = args->mm_data;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  int *mean_vec = args->mean_vec;
  int mat_size_bytes = args->mat_size_bytes;


#ifdef EXETIME
  ts0 = rdtsc();
#endif

  matrix_mult(mm_data);

#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_dsa_memfill_desc_with_preallocated_comp(
        desc, 0,  (uint64_t)mm_data->matrix_out, (uint64_t)comp, mat_size_bytes/2);

  while(enqcmd((void *)((char *)(dsa->wq_reg) + args->p_off), desc) ){ }

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  while(comp->status == IAX_COMP_NONE){  }
  if(comp->status != IAX_COMP_SUCCESS){
    LOG_PRINT(LOG_ERR, "Memfill failed: %d\n", comp->status);
    return;
  }

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  calc_mean(
    mm_data->matrix_out,
    mean_vec,
    mm_data->matrix_len,
    mm_data->matrix_len
  );

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void matmul_memfill_pca_yielding(fcontext_transfer_t arg){
  mmpc_args_t *args =
    (mmpc_args_t *)arg.data;

  mm_data_t *mm_data = args->mm_data;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  int *mean_vec = args->mean_vec;
  int mat_size_bytes = args->mat_size_bytes;

  scheduler_xfer = arg;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  matrix_mult(mm_data);

#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_dsa_memfill_desc_with_preallocated_comp(
        desc, 0,  (uint64_t)mm_data->matrix_out, (uint64_t)comp, mat_size_bytes/2);

  while(enqcmd((void *)((char *)(dsa->wq_reg) + args->p_off), desc) ){ }

#ifdef EXETIME
  ts2 = rdtsc();
#endif


  ret_args.status = REQUEST_YIELDED;
  fcontext_swap(arg.prev_context, NULL);

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  calc_mean(
    mm_data->matrix_out,
    mean_vec,
    mm_data->matrix_len,
    mm_data->matrix_len
  );

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  ret_args.status = REQUEST_COMPLETED;
  fcontext_swap(arg.prev_context, NULL);
}

void gpcore_do_extract(uint8_t *inp, uint8_t *outp, int low_val, int high_val, uint8_t *aecs){
  uint32_t num_inputs = high_val - low_val;

  uint32_t iaa_filter_flags = 28;

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

  iaa_do_extract(outp, inp, (void *)aecs, num_inputs, iaa_filter_flags);
}


void update_filter_histogram_baseline(fcontext_transfer_t arg){
  ufh_args_t *args =
    (ufh_args_t *)arg.data;

  float *upd_buf = args->upd_buf;
  int *scat_buf = args->scat_buf;
  uint8_t *extracted = args->extracted;
  uint8_t *hist = args->hist;
  int upd_sz, hist_bytes;
  int num_acc = args->num_acc;
  int low_val = args->low_val;
  int high_val = args->high_val;
  int num_extracted = high_val - low_val;
  uint8_t *aecs = args->aecs;

  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;

  uint64_t ts0, ts1, ts2, ts3, ts4;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  scatter_update_inplace_using_indir_array(upd_buf, scat_buf, num_acc);

#ifdef EXETIME
  ts1 = rdtsc();
#endif

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  gpcore_do_extract( (uint8_t *)upd_buf, extracted, low_val, high_val, aecs);

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  calc_hist(extracted, hist, num_extracted, &hist_bytes);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void update_filter_histogram_blocking(fcontext_transfer_t arg){
  ufh_args_t *args =
    (ufh_args_t *)arg.data;

  float *upd_buf = args->upd_buf;
  int *scat_buf = args->scat_buf;
  uint8_t *extracted = args->extracted;
  uint8_t *hist = args->hist;
  int upd_sz, hist_bytes;
  int num_acc = args->num_acc;
  int low_val = args->low_val;
  int high_val = args->high_val;
  int num_extracted = high_val - low_val;
  uint8_t *aecs = args->aecs;

  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;

  uint64_t ts0, ts1, ts2, ts3, ts4;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  scatter_update_inplace_using_indir_array(upd_buf, scat_buf, num_acc);

#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_iaa_filter_desc_with_preallocated_comp(
    desc, (uint64_t)upd_buf, (uint64_t)extracted, (uint64_t)comp, num_extracted);

  while(enqcmd((void *)((char *)(iaa->wq_reg) + args->p_off), desc) ){
    /* retry submit */
  }

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  while(comp->status == IAX_COMP_NONE){  }
  if(comp->status != IAX_COMP_SUCCESS){
    LOG_PRINT(LOG_ERR, "Extract failed: %d\n", comp->status);
    return;
  }

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  calc_hist(extracted, hist, num_extracted, &hist_bytes);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  fcontext_swap(arg.prev_context, NULL);
}

void update_filter_histogram_yielding(fcontext_transfer_t arg){
  ufh_args_t *args =
    (ufh_args_t *)arg.data;

  float *upd_buf = args->upd_buf;
  int *scat_buf = args->scat_buf;
  uint8_t *extracted = args->extracted;
  uint8_t *hist = args->hist;
  int upd_sz, hist_bytes;
  int num_acc = args->num_acc;
  int low_val = args->low_val;
  int high_val = args->high_val;
  int num_extracted = high_val - low_val;
  uint8_t *aecs = args->aecs;

  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;

  uint64_t ts0, ts1, ts2, ts3, ts4;

  scheduler_xfer = arg;

#ifdef EXETIME
  ts0 = rdtsc();
#endif

  scatter_update_inplace_using_indir_array(upd_buf, scat_buf, num_acc);

#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_iaa_filter_desc_with_preallocated_comp(
    desc, (uint64_t)upd_buf, (uint64_t)extracted, (uint64_t)comp, num_extracted);

  while(enqcmd((void *)((char *)(iaa->wq_reg) + args->p_off), desc) ){
    /* retry submit */
  }

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  ret_args.status = REQUEST_YIELDED;
  fcontext_swap(arg.prev_context, NULL);

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  calc_hist(extracted, hist, num_extracted, &hist_bytes);

#ifdef EXETIME
  ts4 = rdtsc();
#endif

#ifdef EXETIME
  *args->ts0 = ts0;
  *args->ts1 = ts1;
  *args->ts2 = ts2;
  *args->ts3 = ts3;
  *args->ts4 = ts4;
#endif

  ret_args.status = REQUEST_COMPLETED;
  fcontext_swap(arg.prev_context, NULL);
}


bool gDebugParam = false;
int gLogLevel = LOG_DEBUG;
int *glob_indir_arr = NULL; // TODO
int num_accesses = 0; // TODO
extern struct acctest_context *iaa;
int main(int argc, char **argv){
  /* get arg length for generic executor */
  fcontext_fn_t m_rq_fn = NULL;
  int arg_len = 0;
  char *m_args = NULL;
  int rc;
  uint64_t total_requests = 10;
  int iter = 10;
  #ifdef THROUGHPUT
  uint64_t start, end, *st_times, *end_times, exe_times_avg, *exe_time_diffs;
  #endif

  int nurq = 0;
  int nrq2c = 0;
  int prqidx = -1;
  struct completion_record *dummy;
  fcontext_transfer_t pxfer;
  fcontext_transfer_t yxfer;
  fcontext_transfer_t *offload_req_xfer;
  fcontext_state_t **off_req_state;

  #ifdef SOJOURN
  uint64_t start;
  int indices[] = {0, 1, 2, 3, 4, 5, 10, 100}; /* queue idxs */
  int num_indices = sizeof(indices)/sizeof(indices[0]);

  uint64_t *samples; /* samples for each iter */
  uint64_t *start_dup_array;
  uint64_t **index_samples; /* store times for each queue idx at each iter*/
  uint64_t *sojourn_times;
  #endif
  struct numa_mem  *nm = NULL;
  int nb_numa_node = 1;
  int node = 0;
  idxd_comp *comp = NULL;
  idxd_desc *desc = NULL;
  uint8_t *ser_bufs = NULL;
  char *d_bufs = NULL;
  int pg_size;
  uint64_t i;
  int iaa_dev_id = 1;
  int iaa_wq_id = 0;
  int dsa_dev_id = 0;
  int dsa_wq_id = 0;
  int wq_type = SHARED;
  fcontext_state_t *state = NULL, *fstate = NULL;
  uint8_t *stack_ctxs = NULL;
  uint8_t *f_stack_ctxs = NULL;
  ddh_args_t *ddh_args = NULL;
  uint64_t *ts0, *ts1, *ts2, *ts3, *ts4;
  int cpu = 10;
  a_args_t *f_args = NULL;
  char *f_inp = NULL;
  uint64_t context_size = 16 * 1024;
  uint64_t stack_size = context_size - sizeof(fcontext_state_t);
  uint64_t payload_size = 1 * 1024 * 1024;
  uint64_t filler_payload_size = L3_SIZE;
  double target_ratio = 3.0;
  uint64_t decomp_size = payload_size * target_ratio;
  uint64_t decomp_out_space = IAA_DECOMPRESS_MAX_DEST_SIZE;
  uint64_t max_comp_size = get_compress_bound(decomp_size);
  uint64_t max_payload_expansion = max_comp_size + MAX_SER_OVERHEAD_BYTES;

  char *s_bufs = NULL; /*dg*/
  float *i_bufs = NULL;
  float *o_bufs = NULL;
  int *g_buf = NULL;
  int num_accesses = 0;
  int max_indirect_index = 0;
  dg_args_t *dg_args = NULL;

  /*sa-indep*/
  dl_args_t *dl_args = NULL;
  uint8_t *dsa_src_buf = NULL;
  uint8_t *dsa_dst_buf = NULL;
  uint64_t max_ax_src_sz, avail_ax_dst_size;
  int dl_buf_sz = L1_SIZE;
  uint8_t *dl_bufs = NULL;
  uint8_t *temp_uncomp_buf = NULL;
  int comp_ax_src_size = 0;
  uLong uncomp_ax_src_size = 0;
  max_ax_src_sz = 1024 * 1024;
  avail_ax_dst_size = 1024 * 1024;

  /*sa-dep*/
  int nested_dl_buf_sz = dl_buf_sz;
  uint64_t dl_buf_offset = 0;
  void *nested_dl_buf_start;

  /*decrypt-memcpy-dp */
  dmdp_args_t *dmdp_args = NULL;
  char *enc_bufs = NULL;
  char *dec_bufs = NULL;
  char *tmp_plain_buf = NULL;
  char *cpyd_dec_bufs = NULL;

  /* dsa offload */
  dsa_args_t *dsa_args = NULL;
  char *src_bufs = NULL;
  char *dst_bufs = NULL;
  enum cache_state c_state = L2_DIRTY;
  enum dsa_opcode dsa_opcode = DSA_OPCODE_MEMMOVE;

  /* iaa offload */
  iaa_args_t *iaa_args = NULL;
  char *iaa_src_bufs = NULL;
  char *iaa_dst_bufs = NULL;
  enum iax_opcode iaa_opcode = IAX_OPCODE_DECOMPRESS;
  char *tmp_decomp_buf;
  uLong tmp_decomp_buf_sz = 1024 * 1024;

  /* matmul-memfill-pca */
  mmpc_args_t *mmpc_args = NULL;
  mm_data_t * mm_data = NULL;
  int *mat_a = NULL;
  int *mat_b = NULL;
  int *mat_c = NULL;
  int matrix_len;
  int mat_size_bytes;
  int *mean_vector = NULL;

  /* upd-filt-hist args */
  ufh_args_t *upd_args = NULL;
  float *upd_bufs = NULL;
  uint8_t *extracted_bufs = NULL;
  uint8_t *hist_bufs = NULL;
  int hist_size = 256 * 3 * sizeof(int);
  int *scat_buf = NULL;
  int low_val = 0;
  int high_val = 256;
  uint8_t *aecs = NULL;


  typedef enum _main_request_type_t {
    DESER_DECOMP_HASH,
    DECOMP_GATHER,
    MEMCPY_GATHER,
    SERIAL_ACCESSOR,
    SERIAL_ACCESSOR_DEP,
    SERIAL_ACCESSOR_DECOMP,
    SERIAL_ACCESSOR_DECOMP_DEP,
    DECRYPT_MEMCPY_DP,
    DSA_OFFLOAD,
    IAA_OFFLOAD,
    MATMUL_MEMFILL_PCA,
    UPDATE_FILTER_HISTOGRAM
  } main_request_type_t;

  typedef enum _filler_request_type_t {
    FILLER_INTERRUPTIBLE,
    FILLER_UNINTERRUPTIBLE
  } filler_request_type_t;

  main_request_type_t main_type = DESER_DECOMP_HASH;
  filler_request_type_t filler_type = FILLER_INTERRUPTIBLE;

  // fcontext_state_t *self = fcontext_create_proxy();
  bool just_blocking = false;
  bool just_baseline = false;
  bool neither = false;
  int opt;
  while((opt = getopt(argc, argv, "t:i:s:bgydm:n:k:vbgyc:a:e:l:rf:p:o:jqw")) != -1){
    switch(opt){
      case 'q':
        just_blocking = true;
        break;
      case 'j':
        just_baseline = true;
        break;
      case 'w':
        neither = true;
        break;
      case 's':
        payload_size = atol(optarg);
        break;
      case 't':
        total_requests = atoi(optarg);
        break;
      case 'a':
        main_type = (main_request_type_t)atoi(optarg);
        break;
      case 'l':
        gLogLevel = atoi(optarg);
        break;
      case 'c':
        filler_type = (filler_request_type_t)atoi(optarg);
        break;
      case 'f':
        filler_payload_size = atol(optarg);
      case 'm':
        c_state = (enum cache_state)atoi(optarg);
        break;
      case 'o':
        dsa_opcode = (enum dsa_opcode)atoi(optarg);
        break;
      case 'p':
        c_state = (enum cache_state)atoi(optarg);
        break;
      case 'i':
        iter = atoi(optarg);
        break;
      default:
        break;
    }
  }

  initialize_iaa_wq(iaa_dev_id, iaa_wq_id, wq_type);
  initialize_dsa_wq(dsa_dev_id, dsa_wq_id, wq_type);

  pg_size = 1024 * 1024 * 1024;

  /*decomp-gather / deser-decomp hash / serial- decomp*/
  decomp_size = payload_size * target_ratio;
  decomp_out_space = IAA_DECOMPRESS_MAX_DEST_SIZE;
  max_comp_size = get_compress_bound(decomp_size);
  max_payload_expansion = max_comp_size + MAX_SER_OVERHEAD_BYTES;

  if(main_type == DECOMP_GATHER){
    num_accesses = decomp_size / sizeof(float);
    max_indirect_index = num_accesses - 1;
  } else if(main_type == MEMCPY_GATHER){
    num_accesses = payload_size / sizeof(float);
    max_indirect_index = num_accesses - 1;
  } else if (main_type == UPDATE_FILTER_HISTOGRAM){
    num_accesses = payload_size / sizeof(float);
    max_indirect_index = num_accesses - 1;
  }

  /* upd-filt-hist */
  high_val = payload_size / 2;

  nm = (struct numa_mem *)calloc(nb_numa_node, sizeof(nm[0]));

  comp = (idxd_comp *)alloc_numa_offset(nm, total_requests * sizeof(idxd_comp), 0);
  desc = (idxd_desc *)alloc_numa_offset(nm, total_requests * sizeof(idxd_desc), 0);

  ts0 = (uint64_t *)alloc_numa_offset(nm, total_requests * sizeof(uint64_t), 0);
  ts1 = (uint64_t *)alloc_numa_offset(nm, total_requests * sizeof(uint64_t), 0);
  ts2 = (uint64_t *)alloc_numa_offset(nm, total_requests * sizeof(uint64_t), 0);
  ts3 = (uint64_t *)alloc_numa_offset(nm, total_requests * sizeof(uint64_t), 0);
  ts4 = (uint64_t *)alloc_numa_offset(nm, total_requests * sizeof(uint64_t), 0);

  f_args = (a_args_t *)alloc_numa_offset(nm, total_requests * sizeof(a_args_t), 0);
  f_inp = (char *)alloc_numa_offset(nm, filler_payload_size, 0);
  f_stack_ctxs =
    (uint8_t *)alloc_numa_offset(nm, total_requests * context_size, 0);

  stack_ctxs =
    (uint8_t *)alloc_numa_offset(nm, total_requests * context_size, 0);

  ser_bufs = (uint8_t *)alloc_numa_offset(nm, total_requests * max_payload_expansion, 0);
  d_bufs = (char *)alloc_numa_offset(nm, total_requests * decomp_out_space, 0);

  ddh_args =
    (ddh_args_t *)alloc_numa_offset(nm, total_requests * sizeof(ddh_args_t), 0);

  g_buf = (int *)alloc_numa_offset(nm, num_accesses * sizeof(int), 0);
  s_bufs = (char *)alloc_numa_offset(nm, total_requests * max_comp_size, 0);
  i_bufs = (float *)alloc_numa_offset(nm, total_requests * decomp_out_space, 0);
  dg_args = (dg_args_t *)alloc_numa_offset(nm, total_requests * sizeof(dg_args_t), 0);
  o_bufs = (float *)alloc_numa_offset(nm, total_requests * decomp_out_space, 0);

  dmdp_args = (dmdp_args_t *)alloc_numa_offset(nm, total_requests * sizeof(dmdp_args_t), 0);
  enc_bufs = (char *)alloc_numa_offset(nm, total_requests * payload_size, 0);
  dec_bufs = (char *)alloc_numa_offset(nm, total_requests * payload_size, 0);
  cpyd_dec_bufs = (char *)alloc_numa_offset(nm, total_requests * payload_size, 0);

  dsa_args = (dsa_args_t *)alloc_numa_offset(nm, total_requests * sizeof(dsa_args_t), 0);
  src_bufs = (char *)alloc_numa_offset(nm, total_requests * payload_size, 0);
  dst_bufs = (char *)alloc_numa_offset(nm, total_requests * IAA_DECOMPRESS_MAX_DEST_SIZE, 0); /* just provision for max offload size */

  iaa_args = (iaa_args_t *)alloc_numa_offset(nm, total_requests * sizeof(iaa_args_t), 0);
  iaa_src_bufs = (char *)alloc_numa_offset(nm, total_requests * max_comp_size, 0);
  iaa_dst_bufs = (char *)alloc_numa_offset(nm, total_requests * IAA_DECOMPRESS_MAX_DEST_SIZE, 0); /* just provision for max offload size */

  matrix_len = floor(sqrt(payload_size/sizeof(int)));
  mat_size_bytes = matrix_len * matrix_len * sizeof(int);
  mmpc_args = (mmpc_args_t *)alloc_numa_offset(nm, total_requests * sizeof(mmpc_args_t), 0);
  mm_data = (mm_data_t *)alloc_numa_offset(nm,sizeof(mm_data_t) * total_requests, 0);
  mat_a = (int *)alloc_numa_offset(nm, total_requests * mat_size_bytes, 0);
  mat_b = (int *)alloc_numa_offset(nm, total_requests * mat_size_bytes, 0);
  mat_c = (int *)alloc_numa_offset(nm, total_requests * mat_size_bytes, 0);
  mean_vector = (int *)alloc_numa_offset(nm, total_requests * matrix_len * sizeof(int), 0);

  upd_args = (ufh_args_t *)alloc_numa_offset(nm, total_requests * sizeof(ufh_args_t), 0);
  upd_bufs = (float *)alloc_numa_offset(nm, total_requests * payload_size, 0);
  extracted_bufs = (uint8_t *)alloc_numa_offset(nm, total_requests * IAA_FILTER_MAX_DEST_SIZE, 0);
  hist_bufs = (uint8_t *)alloc_numa_offset(nm, total_requests * hist_size, 0);
  scat_buf = (int *)alloc_numa_offset(nm, num_accesses * sizeof(int), 0);
  aecs = (uint8_t *)alloc_numa_offset(nm, IAA_FILTER_AECS_SIZE * 2 * total_requests, 0); /* enough space for each to get its own cache line*/

  switch(main_type){
    case SERIAL_ACCESSOR_DECOMP_DEP:
    case SERIAL_ACCESSOR_DECOMP:
      if(payload_size < dl_buf_sz){
        nested_dl_buf_sz = decomp_size; // used for both indep and dep to ensure same payload is decomp'd
      }
      max_ax_src_sz = max_comp_size;
      avail_ax_dst_size = decomp_out_space;
      break;
    case SERIAL_ACCESSOR_DEP:
      if(payload_size < dl_buf_sz){
        nested_dl_buf_sz = payload_size;
      }
    case SERIAL_ACCESSOR:
      max_ax_src_sz = payload_size;
      avail_ax_dst_size = payload_size;
      break;
    default:
      break;
  }

  dl_args = (dl_args_t *)alloc_numa_offset(nm, total_requests * sizeof(dl_args_t), 0);
  dsa_src_buf = (uint8_t *)alloc_numa_offset(nm, max_ax_src_sz * total_requests, 0);
  dsa_dst_buf = (uint8_t *)alloc_numa_offset(nm, avail_ax_dst_size * total_requests, 0);
  dl_bufs = (uint8_t *)alloc_numa_offset(nm, total_requests * dl_buf_sz, 0);


  rc = alloc_numa_mem(nm, pg_size, node);
  if(rc){
    PRINT("Failed to allocate memory\n");
    return -1;
  }

  add_base_addr(nm, (void **)&comp);
  add_base_addr(nm, (void **)&desc);
  add_base_addr(nm, (void **)&stack_ctxs);
  add_base_addr(nm, (void **)&ts0);
  add_base_addr(nm, (void **)&ts1);
  add_base_addr(nm, (void **)&ts2);
  add_base_addr(nm, (void **)&ts3);
  add_base_addr(nm, (void **)&ts4);

  add_base_addr(nm, (void **)&f_args);
  add_base_addr(nm, (void **)&f_inp);
  add_base_addr(nm, (void **)&f_stack_ctxs);

  add_base_addr(nm, (void **)&ddh_args);
  add_base_addr(nm, (void **)&ser_bufs);
  add_base_addr(nm, (void **)&d_bufs);

  add_base_addr(nm, (void **)&g_buf);
  add_base_addr(nm, (void **)&s_bufs);
  add_base_addr(nm, (void **)&i_bufs);
  add_base_addr(nm, (void **)&dg_args);
  add_base_addr(nm, (void **)&o_bufs);

  add_base_addr(nm, (void **)&dsa_src_buf);
  add_base_addr(nm, (void **)&dsa_dst_buf);
  add_base_addr(nm, (void **)&dl_bufs);
  add_base_addr(nm, (void **)&dl_args);

  add_base_addr(nm, (void **)&dmdp_args);
  add_base_addr(nm, (void **)&enc_bufs);
  add_base_addr(nm, (void **)&dec_bufs);
  add_base_addr(nm, (void **)&cpyd_dec_bufs);

  add_base_addr(nm, (void **)&src_bufs);
  add_base_addr(nm, (void **)&dst_bufs);
  add_base_addr(nm, (void **)&dsa_args);

  add_base_addr(nm, (void **)&iaa_args);
  add_base_addr(nm, (void **)&iaa_src_bufs);
  add_base_addr(nm, (void **)&iaa_dst_bufs);

  add_base_addr(nm, (void **)&mmpc_args);
  add_base_addr(nm, (void **)&mm_data);
  add_base_addr(nm, (void **)&mat_a);
  add_base_addr(nm, (void **)&mat_b);
  add_base_addr(nm, (void **)&mat_c);
  add_base_addr(nm, (void **)&mean_vector);

  add_base_addr(nm, (void **)&upd_args);
  add_base_addr(nm, (void **)&upd_bufs);
  add_base_addr(nm, (void **)&extracted_bufs);
  add_base_addr(nm, (void **)&hist_bufs);
  add_base_addr(nm, (void **)&scat_buf);
  add_base_addr(nm, (void **)&aecs);


  memset(comp, 0, total_requests * sizeof(idxd_comp));
  memset(desc, 0, total_requests * sizeof(idxd_desc));

  /* all the payloads will match - just allocate once and copy */
  switch(main_type){
    case DESER_DECOMP_HASH:
      for(i=0; i<total_requests; i++){
        if(i == 0){
          gen_ser_comp_payload(
            (void *)(ser_bufs + (i * max_payload_expansion)),
            payload_size, max_comp_size,
            max_payload_expansion,
            &ddh_args[i].s_sz,
            target_ratio);
        } else {
          memcpy((void *)(ser_bufs + (i * max_payload_expansion)), ser_bufs, max_payload_expansion);
        }
        ddh_args[i].s_sz = ddh_args[0].s_sz;
        ddh_args[i].desc = &desc[i];
        ddh_args[i].comp = &comp[i];
        ddh_args[i].d_sz = decomp_size;
        ddh_args[i].d_buf = &d_bufs[i * decomp_out_space];
        ddh_args[i].s_buf = (char *)(ser_bufs + (i * max_payload_expansion));

        ddh_args[i].p_off = (i * 64) % 4096;
        ddh_args[i].ts0 = &ts0[i];
        ddh_args[i].ts1 = &ts1[i];
        ddh_args[i].ts2 = &ts2[i];
        ddh_args[i].ts3 = &ts3[i];
        ddh_args[i].ts4 = &ts4[i];
      }
      break;
    case MEMCPY_GATHER:
      for(i=0; i<total_requests; i++){
        create_random_chain_starting_at(
          payload_size,
          (void **)(s_bufs + (i * payload_size)),
          (void **)(i_bufs + (i * payload_size/sizeof(float)))
        );
        dg_args[i].s_sz = payload_size;
        dg_args[i].desc = &desc[i];
        dg_args[i].comp = &comp[i];
        dg_args[i].d_sz = payload_size;
        dg_args[i].d_buf = &i_bufs[i * (payload_size/sizeof(float))]; // float arr
        dg_args[i].s_buf = (char *)&s_bufs[i * payload_size]; // char arr
        dg_args[i].g_buf = g_buf;
        dg_args[i].o_buf = &o_bufs[i * (payload_size/sizeof(float))]; // float arr
        dg_args[i].num_accesses = num_accesses;

        dg_args[i].p_off = (i * 64) % 4096;
        dg_args[i].ts0 = &ts0[i];
        dg_args[i].ts1 = &ts1[i];
        dg_args[i].ts2 = &ts2[i];
        dg_args[i].ts3 = &ts3[i];
        dg_args[i].ts4 = &ts4[i];
      }
      break;
    case DECOMP_GATHER:
      temp_uncomp_buf = (uint8_t *)malloc(decomp_size);
      lzdg_generate_reuse_buffers(temp_uncomp_buf, decomp_size, 3.0, 3.0, 3.0);
      for(i=0; i<total_requests; i++){
        create_random_chain_starting_at(
          decomp_size,
          (void **)(temp_uncomp_buf),
          (void **)&(i_bufs[i * (decomp_out_space/sizeof(float))])
        );
        comp_ax_src_size = max_comp_size;
        gpcore_do_compress(
          (void *)&(s_bufs[i * max_comp_size]),
          (void *)temp_uncomp_buf,
          decomp_size,
          &comp_ax_src_size);

        if(i == 0){
          PRINT("Payload size: %lu\n", payload_size);
          PRINT("Comp size: %d\n", comp_ax_src_size);
          PRINT("Decomp size: %lu\n", decomp_size);

        }
        dg_args[i].s_sz = comp_ax_src_size;
        dg_args[i].desc = &desc[i];
        dg_args[i].comp = &comp[i];
        dg_args[i].d_sz = decomp_size;
        dg_args[i].d_buf = &i_bufs[i * (decomp_out_space/sizeof(float))]; // float arr
        dg_args[i].s_buf = (char *)&s_bufs[i * max_comp_size]; // char arr
        dg_args[i].g_buf = g_buf;
        dg_args[i].o_buf = &o_bufs[i * (decomp_out_space/sizeof(float))]; // float arr
        dg_args[i].num_accesses = num_accesses;

        dg_args[i].p_off = (i * 64) % 4096;
        dg_args[i].ts0 = &ts0[i];
        dg_args[i].ts1 = &ts1[i];
        dg_args[i].ts2 = &ts2[i];
        dg_args[i].ts3 = &ts3[i];
        dg_args[i].ts4 = &ts4[i];
      }
      free(temp_uncomp_buf);
      break;
    case SERIAL_ACCESSOR:
      for(i=0; i < total_requests * dl_buf_sz; i+= dl_buf_sz){
        create_random_chain_in_place(dl_buf_sz, (void **)&dl_bufs[i]);
      }
      for(i=0; i<total_requests; i++){
        dl_args[i].desc = &desc[i];
        dl_args[i].comp = &comp[i];
        dl_args[i].src = (void *)&(dsa_src_buf[i * max_ax_src_sz]);
        dl_args[i].dst = (void *)&(dsa_dst_buf[i * max_ax_src_sz]);
        dl_args[i].dl_buf = (void **)&dl_bufs[i * dl_buf_sz];
        dl_args[i].xfer_size = max_ax_src_sz;
        dl_args[i].dl_size = dl_buf_sz;
        dl_args[i].p_off = (i * 64) % 4096;
        dl_args[i].ts0 = &ts0[i];
        dl_args[i].ts1 = &ts1[i];
        dl_args[i].ts2 = &ts2[i];
        dl_args[i].ts3 = &ts3[i];
        dl_args[i].ts4 = &ts4[i];
      }
      break;
    case SERIAL_ACCESSOR_DEP:
      for(i=0; i< total_requests; i++){
        dl_args[i].dl_buf = (void **)&dl_bufs[i * dl_buf_sz];
        dl_buf_offset = payload_size - nested_dl_buf_sz;
        nested_dl_buf_start = (void *)(((char *)&dsa_src_buf[i * max_ax_src_sz]) + dl_buf_offset);
        create_random_chain_starting_at(
          nested_dl_buf_sz,
          (void **)(nested_dl_buf_start),
          (void **)&dsa_dst_buf[(i * max_ax_src_sz) + dl_buf_offset]);
        dl_args[i].src = (void *)&(dsa_src_buf[i * max_ax_src_sz]);
        dl_args[i].dst = (void *)&(dsa_dst_buf[i * max_ax_src_sz]);
        dl_args[i].desc = &desc[i];
        dl_args[i].comp = &comp[i];
        dl_args[i].xfer_size = max_ax_src_sz;
        dl_args[i].dl_size = nested_dl_buf_sz;
        dl_args[i].p_off = (i * 64) % 4096;
        dl_args[i].ts0 = &ts0[i];
        dl_args[i].ts1 = &ts1[i];
        dl_args[i].ts2 = &ts2[i];
        dl_args[i].ts3 = &ts3[i];
        dl_args[i].ts4 = &ts4[i];
      }
      break;
    case SERIAL_ACCESSOR_DECOMP_DEP:
    case SERIAL_ACCESSOR_DECOMP:
      if(temp_uncomp_buf == NULL){
        temp_uncomp_buf = (uint8_t *)malloc(decomp_size);
        lzdg_generate_data((void *)temp_uncomp_buf, decomp_size, 3.0, 3.0, 3.0);
      } else {
        PRINT("Error: temp_uncomp_buf already allocated\n");
        exit(1);
      }
      dl_buf_offset = decomp_size - nested_dl_buf_sz;
      nested_dl_buf_start = (void *)(temp_uncomp_buf + dl_buf_offset);
      for(i=0; i < total_requests * dl_buf_sz; i+= dl_buf_sz){
        create_random_chain_in_place(dl_buf_sz, (void **)&dl_bufs[i]);
      }
      for(i=0; i< total_requests; i++){
        dl_args[i].dl_buf = (void **)&dl_bufs[i * dl_buf_sz];
        create_random_chain_starting_at(nested_dl_buf_sz,
          (void **)(nested_dl_buf_start),(void **)&dsa_dst_buf[(i * avail_ax_dst_size) + dl_buf_offset]);
        comp_ax_src_size = max_ax_src_sz; /* avail out */
        gpcore_do_compress(
          (void *)&(dsa_src_buf[i * max_ax_src_sz]),
          (void *)temp_uncomp_buf,
          decomp_size,
          &comp_ax_src_size);
        dl_args[i].src = (void *)&(dsa_src_buf[i * max_ax_src_sz]);
        dl_args[i].dst = (void *)&(dsa_dst_buf[i * avail_ax_dst_size]);
        dl_args[i].desc = &desc[i];
        dl_args[i].comp = &comp[i];
        dl_args[i].xfer_size = (uint64_t )comp_ax_src_size;
        dl_args[i].uncomp_size = decomp_size;
        if(main_type == SERIAL_ACCESSOR_DECOMP_DEP){
          dl_args[i].dl_size = nested_dl_buf_sz;
        } else {
          dl_args[i].dl_size = dl_buf_sz;
        }
        dl_args[i].p_off = (i * 64) % 4096;
        dl_args[i].ts0 = &ts0[i];
        dl_args[i].ts1 = &ts1[i];
        dl_args[i].ts2 = &ts2[i];
        dl_args[i].ts3 = &ts3[i];
        dl_args[i].ts4 = &ts4[i];
      }
      free(temp_uncomp_buf);
      break;
    case DECRYPT_MEMCPY_DP:
      for(i=0; i<total_requests; i++){
        if(i==0){
          tmp_plain_buf = (char *)malloc(payload_size);
          memset_pattern((void *)tmp_plain_buf, 0xdeadbeef, payload_size);
          enc_buf((Ipp8u *)&enc_bufs[i *payload_size], (Ipp8u *) tmp_plain_buf, payload_size);
        } else {
          memcpy((void *)&enc_bufs[i * payload_size], (void *)&enc_bufs[0], payload_size);
        }
        dmdp_args[i].sz = payload_size;
        dmdp_args[i].desc = &desc[i];
        dmdp_args[i].comp = &comp[i];
        dmdp_args[i].dec_buf = &dec_bufs[i * payload_size]; // char arr
        dmdp_args[i].enc_buf = (char *)&enc_bufs[i * payload_size]; // char arr
        dmdp_args[i].dst_buf = (char *)&cpyd_dec_bufs[i * payload_size]; // char arr
        dmdp_args[i].p_off = (i * 64) % 4096;
        dmdp_args[i].ts0 = &ts0[i];
        dmdp_args[i].ts1 = &ts1[i];
        dmdp_args[i].ts2 = &ts2[i];
        dmdp_args[i].ts3 = &ts3[i];
        dmdp_args[i].ts4 = &ts4[i];
      }

      break;
    case DSA_OFFLOAD:
      for(i=0; i<total_requests; i++){
        dsa_args[i].c_state = c_state;
        dsa_args[i].opcode = dsa_opcode;
        dsa_args[i].src = &src_bufs[i * payload_size];
        dsa_args[i].dst = &dst_bufs[i * IAA_DECOMPRESS_MAX_DEST_SIZE];
        dsa_args[i].desc = &desc[i];
        dsa_args[i].comp = &comp[i];
        dsa_args[i].xfer_size = payload_size;
        dsa_args[i].p_off = (i * 64) % 4096;
        dsa_args[i].ts0 = &ts0[i];
        dsa_args[i].ts1 = &ts1[i];
        dsa_args[i].ts2 = &ts2[i];
        dsa_args[i].ts3 = &ts3[i];
        dsa_args[i].ts4 = &ts4[i];
        for(int j=0; j<IAA_DECOMPRESS_MAX_DEST_SIZE; j+=4096){
          dsa_args[i].dst[j] = 'a';
        }
      }
      break;
    case IAA_OFFLOAD:
      switch(iaa_opcode){
        case IAX_OPCODE_DECOMPRESS:
          gen_comp_buf(payload_size, max_comp_size,
              (void *)&iaa_src_bufs[i * max_comp_size], &(iaa_args[0].xfer_size), target_ratio);
          for(int i=0; i<total_requests; i++){
            memcpy((void *)&iaa_src_bufs[i * max_comp_size], (void *)&iaa_src_bufs[0], max_comp_size);
            iaa_args[i].opcode = iaa_opcode;
            iaa_args[i].src = &iaa_src_bufs[i * max_comp_size];
            iaa_args[i].dst = &iaa_dst_bufs[i * IAA_DECOMPRESS_MAX_DEST_SIZE];
            iaa_args[i].desc = &desc[i];
            iaa_args[i].comp = &comp[i];
            iaa_args[i].xfer_size = iaa_args[0].xfer_size;
            iaa_args[i].c_state  = c_state;
            iaa_args[i].p_off = (i * 64) % 4096;
            iaa_args[i].ts0 = &ts0[i];
            iaa_args[i].ts1 = &ts1[i];
            iaa_args[i].ts2 = &ts2[i];
            iaa_args[i].ts3 = &ts3[i];
            iaa_args[i].ts4 = &ts4[i];
          }
          tmp_decomp_buf = (char *)malloc(IAA_DECOMPRESS_MAX_DEST_SIZE);
          tmp_decomp_buf_sz = IAA_DECOMPRESS_MAX_DEST_SIZE;
          break;
        default:
          break;
      }
      break;
    case MATMUL_MEMFILL_PCA:
      for(i=0; i<total_requests; i++){
        mmpc_args[i].desc = &desc[i];
        mmpc_args[i].comp = &comp[i];
        mmpc_args[i].mm_data = (mm_data_t *)&mm_data[i];
        mmpc_args[i].mm_data->matrix_A = &mat_a[i * mat_size_bytes];
        mmpc_args[i].mm_data->matrix_B = &mat_b[i * mat_size_bytes];
        mmpc_args[i].mm_data->matrix_out = &mat_c[i * mat_size_bytes];
        memset(mmpc_args[i].mm_data->matrix_out, 0, mat_size_bytes);
        mmpc_args[i].mat_size_bytes = mat_size_bytes;
          /* two identity matrices */
        for(int k=0; k<matrix_len; k++){
          for(int j=0; j<matrix_len; j++){
            if(k == j){
              mmpc_args[i].mm_data->matrix_A[k * matrix_len + j] = 1;
              mmpc_args[i].mm_data->matrix_B[k * matrix_len + j] = 1;
            } else {
              mmpc_args[i].mm_data->matrix_A[k * matrix_len + j] = 0;
              mmpc_args[i].mm_data->matrix_B[k * matrix_len + j] = 0;
            }
          }
        }
        mmpc_args[i].mm_data->matrix_len = matrix_len;
        mmpc_args[i].mean_vec = &mean_vector[i * matrix_len];
        mmpc_args[i].p_off = (i * 64) % 4096;
        mmpc_args[i].ts0 = &ts0[i];
        mmpc_args[i].ts1 = &ts1[i];
        mmpc_args[i].ts2 = &ts2[i];
        mmpc_args[i].ts3 = &ts3[i];
        mmpc_args[i].ts4 = &ts4[i];
      }
      break;
    case UPDATE_FILTER_HISTOGRAM:
      indirect_array_populate(scat_buf, num_accesses, 0, max_indirect_index);
      for(int i=0; i<total_requests; i++){
        memset_pattern((void *)&upd_bufs[i * payload_size], 0xabcdbeef, payload_size);
        memset((void *)&upd_bufs[i * payload_size], 10, payload_size);
        upd_args[i].desc = &desc[i];
        upd_args[i].comp = &comp[i];
        upd_args[i].upd_buf = &upd_bufs[i * payload_size];
        upd_args[i].extracted = &extracted_bufs[i * IAA_FILTER_MAX_DEST_SIZE];
        upd_args[i].hist = &hist_bufs[i * hist_size];
        upd_args[i].scat_buf = scat_buf;
        upd_args[i].low_val = low_val;
        upd_args[i].high_val = high_val;
        upd_args[i].aecs =&aecs[i * (IAA_FILTER_AECS_SIZE * 2) ];
        upd_args[i].num_acc = num_accesses;

        upd_args[i].p_off = (i * 64) % 4096;
        upd_args[i].ts0 = &ts0[i];
        upd_args[i].ts1 = &ts1[i];
        upd_args[i].ts2 = &ts2[i];
        upd_args[i].ts3 = &ts3[i];
        upd_args[i].ts4 = &ts4[i];

      }
      break;
    default:
      break;
  }

  cpu_pin(cpu);


  uint64_t avg, diff[total_requests];
  uint64_t pre_proc_times;
  uint64_t offload_tax_times;
  uint64_t ax_func_times;
  uint64_t post_proc_times;

  /* Used for Yielding*/
  int l_fidx = 0;
  bool new_filler = true;
  fcontext_transfer_t f_xfer;

  // goto baseline;
  if(neither){
    return 0;
  } else if(just_blocking){
    goto blocking;
  }

  baseline:

  #ifdef THROUGHPUT
  st_times = (uint64_t *) malloc(sizeof(uint64_t) * iter);
  end_times = (uint64_t *)malloc(sizeof(uint64_t) * iter);
  exe_time_diffs = (uint64_t *)malloc(sizeof(uint64_t) * iter);

  for(int k=0; k<iter; k++){
  #endif

  #ifdef SOJOURN
  samples = (uint64_t *)malloc(sizeof(uint64_t) * total_requests);
  index_samples = (uint64_t **)malloc(sizeof(uint64_t*) * num_indices);
  sojourn_times = (uint64_t *)malloc(sizeof(uint64_t) * iter);
  start_dup_array = (uint64_t *)malloc(sizeof(uint64_t) * iter);
  for(int i=0; i<num_indices; i++){
    index_samples[i] = (uint64_t *)malloc(sizeof(uint64_t) * iter);
  }
  for(int k=0; k<iter; k++){
  #endif

  /* gpCore */
  for(i=0; i < total_requests * context_size; i+= context_size){
    state = (fcontext_state_t *)(&stack_ctxs[i]);
    uint8_t *stack_top = state->stack + stack_size;
    switch(main_type){
      case DESER_DECOMP_HASH:
        state->context =
          make_fcontext(stack_top,
                        stack_size, deser_decomp_hash_baseline);
        break;
      case DECOMP_GATHER:
        state->context =
          make_fcontext(stack_top,
                        stack_size, decomp_gather_baseline);
        break;
      case MEMCPY_GATHER:
        state->context =
          make_fcontext(stack_top,
                        stack_size, memcpy_gather_baseline);
        break;
      case SERIAL_ACCESSOR:
        state->context =
          make_fcontext(stack_top,
                        stack_size, serial_accessor_dsa_memcpy_baseline);
        break;
      case SERIAL_ACCESSOR_DEP:
        state->context =
          make_fcontext(stack_top,
                        stack_size, serial_accessor_axdep_dsa_memcpy_baseline);
        break;
      case SERIAL_ACCESSOR_DECOMP_DEP:
        state->context =
          make_fcontext(stack_top,
                        stack_size, serial_accessor_axdep_iaa_decompress_baseline);

          break;

      case DECRYPT_MEMCPY_DP:
        state->context =
          make_fcontext(stack_top,
                        stack_size, decrypt_memcpy_dotproduct_baseline);
        break;
      case MATMUL_MEMFILL_PCA:
        state->context =
          make_fcontext(stack_top,
                        stack_size, matmul_memfill_pca_baseline);
        break;
      case UPDATE_FILTER_HISTOGRAM:
        state->context =
          make_fcontext(stack_top,
                        stack_size, update_filter_histogram_baseline);
        break;
      default:
        break;
    }
  }

  if(main_type == MEMCPY_GATHER){
    flush_range((void *)s_bufs, payload_size * total_requests);
  } else if (main_type == DECOMP_GATHER){
    flush_range((void *)s_bufs, max_comp_size * total_requests);
  }

  #ifdef THROUGHPUT
    start = rdtsc();
  #endif

  #ifdef SOJOURN
    start = rdtsc();
  #endif

  switch(main_type){
    case DESER_DECOMP_HASH:
      m_rq_fn = deser_decomp_hash_baseline;
      arg_len = sizeof(ddh_args_t);
      m_args = (char *)ddh_args;
      break;
    case DECOMP_GATHER:
      m_rq_fn = decomp_gather_baseline;
      arg_len = sizeof(dg_args_t);
      m_args = (char *)dg_args;
      break;
    case MEMCPY_GATHER:
      m_rq_fn = memcpy_gather_baseline;
      arg_len = sizeof(dg_args_t);
      m_args = (char *)dg_args;
      break;
    case SERIAL_ACCESSOR:
    case SERIAL_ACCESSOR_DEP:
      for(i=0; i<total_requests; i++){
        state = (fcontext_state_t *)(&stack_ctxs[i * context_size]);
        fcontext_transfer_t child =  fcontext_swap(state->context, (void *)&dl_args[i]);
      }
      break;
    case SERIAL_ACCESSOR_DECOMP:
    case SERIAL_ACCESSOR_DECOMP_DEP:
      for(i=0; i<total_requests; i++){
        state = (fcontext_state_t *)(&stack_ctxs[i * context_size]);
        fcontext_transfer_t child =  fcontext_swap(state->context, (void *)&dl_args[i]);
      }
      break;
    case DECRYPT_MEMCPY_DP:
      m_rq_fn = decrypt_memcpy_dotproduct_baseline;
      arg_len = sizeof(dmdp_args_t);
      m_args = (char *)dmdp_args;
      break;
    case MATMUL_MEMFILL_PCA:
      m_rq_fn = matmul_memfill_pca_baseline;
      arg_len = sizeof(mmpc_args_t);
      m_args = (char *)mmpc_args;
      break;
    case UPDATE_FILTER_HISTOGRAM:
      m_rq_fn = update_filter_histogram_baseline;
      arg_len = sizeof(ufh_args_t);
      m_args = (char *)upd_args;
      break;
    default:
      break;
  }
  for(i=0; i<total_requests; i++){
    state = (fcontext_state_t *)(&stack_ctxs[i * context_size]);
    fcontext_transfer_t child =  fcontext_swap(state->context, (void *)(m_args + i * arg_len));
    #ifdef SOJOURN
    samples[i] = rdtsc();
    #endif
  }

  #ifdef THROUGHPUT
    end = rdtsc();
    st_times[k] = start;
    end_times[k] = end;
  }
  avg_samples_from_arrays(exe_time_diffs, exe_times_avg, end_times, st_times, iter);
  PRINT("NoAcceleration ");
  PRINT(" %lu", payload_size);
  mean_median_stdev_rps(exe_time_diffs, iter, total_requests, " RPS");

  #endif

  #ifdef SOJOURN

  /* need to track end times for each of these indices */
  for(int i=0; i<num_indices; i++){
      int j = indices[i];
      index_samples[i][k] = samples[j];
    }
    start_dup_array[k] = start;
  }

  PRINT("PayloadSize QueueIdx ModeOfExe SojournTime\n");
  for(int i=0; i<num_indices; i++){
    int j = indices[i];
    avg_samples_from_arrays(sojourn_times, avg, index_samples[i], start_dup_array, iter);
    PRINT("%lu %d %s %lu\n", payload_size, j, "NoAcceleration", avg);
  }
  #endif



	#ifdef EXETIME
  PRINT("ModeOfExe PreProc OffloadTax AxFunc PostProc\n");
  PRINT("Baseline ");
  PRINT(" %lu", payload_size);

  avg_samples_from_arrays(diff, pre_proc_times, ts1, ts0, total_requests);
  PRINT( " %lu", pre_proc_times);

  avg_samples_from_arrays(diff, offload_tax_times, ts2, ts1, total_requests);
  PRINT( " %lu", offload_tax_times);

  avg_samples_from_arrays(diff, ax_func_times, ts3, ts2, total_requests);
  PRINT( " %lu", ax_func_times);

  avg_samples_from_arrays(diff, post_proc_times, ts4, ts3, total_requests);
  PRINT( " %lu\n", post_proc_times);
  #endif

  /* blocking */
blocking:

if(just_baseline){
  return 0;
}

  memset(comp, 0, total_requests * sizeof(idxd_comp));
  memset(desc, 0, total_requests * sizeof(idxd_desc));

  #ifdef THROUGHPUT
  st_times = (uint64_t *) malloc(sizeof(uint64_t) * iter);
  end_times = (uint64_t *)malloc(sizeof(uint64_t) * iter);
  exe_time_diffs = (uint64_t *)malloc(sizeof(uint64_t) * iter);

  for(int k=0; k<iter; k++){
  #endif

  #ifdef SOJOURN
  for(int k=0; k<iter; k++){
  #endif

  for(i=0; i < total_requests * context_size; i+= context_size){
    state = (fcontext_state_t *)(&stack_ctxs[i]);
    uint8_t *stack_top = state->stack + stack_size;
    switch(main_type){
      case DESER_DECOMP_HASH:
        state->context =
          make_fcontext(stack_top,
                        stack_size, deser_decomp_hash_blocking);
        break;
      case DECOMP_GATHER:
        state->context =
          make_fcontext(stack_top,
                        stack_size, decomp_gather_blocking);
        break;
      case MEMCPY_GATHER:
        state->context =
          make_fcontext(stack_top,
                        stack_size, memcpy_gather_blocking);
        break;
      case SERIAL_ACCESSOR:
        state->context =
          make_fcontext(stack_top,
                        stack_size, serial_accessor_dsa_memcpy_blocking);
        break;
      case SERIAL_ACCESSOR_DEP:
        state->context =
          make_fcontext(stack_top,
                        stack_size, serial_accessor_axdep_dsa_memcpy_blocking);
        break;
      case SERIAL_ACCESSOR_DECOMP:
        state->context =
          make_fcontext(stack_top,
                        stack_size, serial_accessor_iaa_decompress_blocking);
        break;
      case SERIAL_ACCESSOR_DECOMP_DEP:
        state->context =
          make_fcontext(stack_top,
                        stack_size, serial_accessor_axdep_iaa_decompress_blocking);
        break;
      case DECRYPT_MEMCPY_DP:
        state->context =
          make_fcontext(stack_top,
                        stack_size, decrypt_memcpy_dotproduct_blocking);
        break;
      case DSA_OFFLOAD:
        state->context =
          make_fcontext(stack_top,
                        stack_size, dsa_offload);
        break;
      case IAA_OFFLOAD:
        state->context =
          make_fcontext(stack_top,
                        stack_size, iaa_offload);
        break;
      case MATMUL_MEMFILL_PCA:
        state->context =
          make_fcontext(stack_top,
                        stack_size, matmul_memfill_pca_blocking);
        break;
      case UPDATE_FILTER_HISTOGRAM:
        state->context =
          make_fcontext(stack_top,
                        stack_size, update_filter_histogram_blocking);
        break;
      default:
        break;
    }
  }

  if(main_type == MEMCPY_GATHER){
    flush_range((void *)s_bufs, payload_size * total_requests);
  } else if (main_type == DECOMP_GATHER){
    flush_range((void *)s_bufs, max_comp_size * total_requests);
  }

  #ifdef THROUGHPUT
    start = rdtsc();
  #endif
  #ifdef SOJOURN
    start = rdtsc();
  #endif

  switch(main_type){
    case DESER_DECOMP_HASH:
        m_rq_fn = deser_decomp_hash_blocking;
        arg_len = sizeof(ddh_args_t);
        m_args = (char *)ddh_args;
        break;
      break;
    case DECOMP_GATHER:
      m_rq_fn = decomp_gather_baseline;
      arg_len = sizeof(dg_args_t);
      m_args = (char *)dg_args;
      break;
    case MEMCPY_GATHER:
      m_rq_fn = memcpy_gather_baseline;
      arg_len = sizeof(dg_args_t);
      m_args = (char *)dg_args;
      break;
    case SERIAL_ACCESSOR:
    case SERIAL_ACCESSOR_DEP:
    case SERIAL_ACCESSOR_DECOMP:
    case SERIAL_ACCESSOR_DECOMP_DEP:
      for(i=0; i<total_requests; i++){
        state = (fcontext_state_t *)(&stack_ctxs[i * context_size]);
        fcontext_transfer_t child =  fcontext_swap(state->context, (void *)&dl_args[i]);
      }
      break;
    case DECRYPT_MEMCPY_DP:
      m_rq_fn = decrypt_memcpy_dotproduct_blocking;
      arg_len = sizeof(dmdp_args_t);
      m_args = (char *)dmdp_args;
      break;
    case IAA_OFFLOAD:
      for(i=0; i<total_requests; i++){
        state = (fcontext_state_t *)(&stack_ctxs[i * context_size]);
        fcontext_transfer_t child =  fcontext_swap(state->context, (void *)&iaa_args[i]);
      }
      break;
    case DSA_OFFLOAD:
      for(i=0; i<total_requests; i++){
        state = (fcontext_state_t *)(&stack_ctxs[i * context_size]);
        fcontext_transfer_t child =  fcontext_swap(state->context, (void *)&dsa_args[i]);
      }
      break;
    case MATMUL_MEMFILL_PCA:
      m_rq_fn = matmul_memfill_pca_blocking;
      arg_len = sizeof(mmpc_args_t);
      m_args = (char *)mmpc_args;
      break;
    case UPDATE_FILTER_HISTOGRAM:
      m_rq_fn = update_filter_histogram_blocking;
      arg_len = sizeof(ufh_args_t);
      m_args = (char *)upd_args;
      break;
    default:
      break;
  }

  for(i=0; i<total_requests; i++){
    state = (fcontext_state_t *)(&stack_ctxs[i * context_size]);
    fcontext_transfer_t child =  fcontext_swap(state->context, (void *)(m_args + i * arg_len));
    #ifdef SOJOURN
    samples[i] = rdtsc();
    #endif
  }

  #ifdef THROUGHPUT
    end = rdtsc();
    st_times[k] = start;
    end_times[k] = end;
  }
  avg_samples_from_arrays(exe_time_diffs, exe_times_avg, end_times, st_times, iter);
  PRINT("Block&Wait ");
  PRINT(" %lu", payload_size);
  mean_median_stdev_rps(exe_time_diffs, iter, total_requests, " RPS");

  #endif

	#ifdef EXETIME

  PRINT("Blocking ");
  PRINT(" %lu", payload_size);
  avg_samples_from_arrays(diff, pre_proc_times, ts1, ts0, total_requests);
  PRINT( " %lu", pre_proc_times);

  avg_samples_from_arrays(diff, offload_tax_times, ts2, ts1, total_requests);
  PRINT( " %lu", offload_tax_times);

  avg_samples_from_arrays(diff, ax_func_times, ts3, ts2, total_requests);
  PRINT( " %lu", ax_func_times);

  avg_samples_from_arrays(diff, post_proc_times, ts4, ts3, total_requests);
  PRINT( " %lu\n", post_proc_times);
  // return 0;
  #endif

  #ifdef SOJOURN
    /* need to track end times for each of these indices */
    for(int i=0; i<num_indices; i++){
      int j = indices[i];
      index_samples[i][k] = samples[j];
    }
    start_dup_array[k] = start;
  }

  PRINT("PayloadSize QueueIdx ModeOfExe SojournTime\n");
  for(int i=0; i<num_indices; i++){
    int j = indices[i];
    avg_samples_from_arrays(sojourn_times, avg, index_samples[i], start_dup_array, iter);
    PRINT("%lu %d %s %lu\n", payload_size, j, "Block&Wait", avg);
  }
  #endif


  /* yeidling */
  #ifdef THROUGHPUT
  st_times = (uint64_t *) malloc(sizeof(uint64_t) * iter);
  end_times = (uint64_t *)malloc(sizeof(uint64_t) * iter);
  exe_time_diffs = (uint64_t *)malloc(sizeof(uint64_t) * iter);

  for(int k=0; k<iter; k++){
  #endif

  #ifdef SOJOURN
  for(int k=0; k<iter; k++){
  #endif

  for(i=0; i < total_requests * context_size; i+= context_size){
    state = (fcontext_state_t *)(&stack_ctxs[i]);
    fstate = (fcontext_state_t *)(&f_stack_ctxs[i]);
    uint8_t *stack_top = state->stack + stack_size;

    switch(main_type){
      case DESER_DECOMP_HASH:
        state->context =
          make_fcontext(stack_top,
                        stack_size, deser_decomp_hash_yielding);
        break;
      case MEMCPY_GATHER:
        state->context =
          make_fcontext(stack_top,
                        stack_size, memcpy_gather_yielding);
        break;
      case DECOMP_GATHER:
        state->context =
          make_fcontext(stack_top,
                        stack_size, decomp_gather_yielding);
        break;
      case SERIAL_ACCESSOR:
        state->context =
          make_fcontext(stack_top,
                        stack_size, serial_accessor_dsa_memcpy_yielding);
        break;
      case SERIAL_ACCESSOR_DEP:
        state->context =
          make_fcontext(stack_top,
                        stack_size, serial_accessor_axdep_dsa_memcpy_yielding);
        break;
      case SERIAL_ACCESSOR_DECOMP:
        state->context =
          make_fcontext(stack_top,
                        stack_size, serial_accessor_iaa_decompress_yielding);
        break;
      case SERIAL_ACCESSOR_DECOMP_DEP:
        state->context =
          make_fcontext(stack_top,
                        stack_size, serial_accessor_axdep_iaa_decompress_yielding);
        break;
      case DECRYPT_MEMCPY_DP:
        state->context =
          make_fcontext(stack_top,
                        stack_size, decrypt_memcpy_dotproduct_yielding);
        break;
      case MATMUL_MEMFILL_PCA:
        state->context =
          make_fcontext(stack_top,
                        stack_size, matmul_memfill_pca_yielding);
        break;
      case UPDATE_FILTER_HISTOGRAM:
        state->context =
          make_fcontext(stack_top,
                        stack_size, update_filter_histogram_yielding);
        break;
      default:
        break;
    }
    stack_top = fstate->stack + stack_size;

    switch(filler_type){
      case FILLER_INTERRUPTIBLE:
        fstate->context =
          make_fcontext(stack_top,
                        stack_size, filler_cache_evict_interruptible);
        break;
      case FILLER_UNINTERRUPTIBLE:
        fstate->context =
          make_fcontext(stack_top,
                        stack_size, filler_cache_evict_uninterruptible);
        break;
      default:
        break;
    }

  }



  for(i=0; i<total_requests; i++){
    f_args[i].input = f_inp;
    f_args[i].input_size = filler_payload_size;
    f_args[i].done = false;
  }

  #ifdef SOJOURN
  goto yp_same;
  #endif

  /* reset comps and descs at the start of the run */
  memset(comp, 0, total_requests * sizeof(idxd_comp));
  memset(desc, 0, total_requests * sizeof(idxd_desc));
  new_filler = true;


  if(main_type == MEMCPY_GATHER){
    flush_range((void *)s_bufs, payload_size * total_requests);
  } else if (main_type == DECOMP_GATHER){
    flush_range((void *)s_bufs, max_comp_size * total_requests);
  }

  #ifdef THROUGHPUT
    start = rdtsc();
  #endif

  preempt_signal = (uint8_t *)&comp[0];
  switch(main_type){
    case DESER_DECOMP_HASH:
//       for(i=0; i<total_requests; i++){
//         state = (fcontext_state_t *)(&stack_ctxs[i * context_size]);
//         fcontext_transfer_t child =  fcontext_swap(state->context, (void *)&ddh_args[i]);
//         preempt_signal = (uint8_t *)&comp[i];
//         if(!new_filler){
//           f_xfer = fcontext_swap(f_xfer.prev_context, NULL);
//         } else {
// new_filler:
//           fstate = (fcontext_state_t *)(&f_stack_ctxs[l_fidx * context_size]);
//           f_xfer = fcontext_swap(fstate->context, (void *)&f_args[l_fidx]);
//         }
//         if(f_args[l_fidx].done){
//           new_filler = true;
//           l_fidx++;
//           if(comp[i].status == IAX_COMP_NONE){
//             goto new_filler;
//           }
//         } else {
//           new_filler = false;
//         }
//         if(__glibc_likely(comp[i].status == IAX_COMP_SUCCESS)){
//           preempt_signal[0] = 0; /* reset to keep preemptible post-proc from yielding*/
//           fcontext_swap(child.prev_context, NULL);
//         }
//         else {
//           LOG_PRINT(LOG_ERR, "Offload failed: %d\n", comp[i].status);
//           return 0;
//         }

//         if(l_fidx == total_requests){
//           PRINT("Fillers completed\n");
//           break;
//         }
//       }
      break;
    case DECOMP_GATHER:
    case MEMCPY_GATHER:
//       for(i=0; i<total_requests; i++){
//         state = (fcontext_state_t *)(&stack_ctxs[i * context_size]);
//         fcontext_transfer_t child =  fcontext_swap(state->context, (void *)&dg_args[i]);
//         preempt_signal = (uint8_t *)&comp[i];
//         if(!new_filler){
//           f_xfer = fcontext_swap(f_xfer.prev_context, NULL);
//         } else {
// new_filler_1:
//           fstate = (fcontext_state_t *)(&f_stack_ctxs[l_fidx * context_size]);
//           f_xfer = fcontext_swap(fstate->context, (void *)&f_args[l_fidx]);
//         }
//         if(f_args[l_fidx].done){
//           new_filler = true;
//           l_fidx++;
//           if(comp[i].status == IAX_COMP_NONE){
//             goto new_filler_1;
//           }
//         } else {
//           new_filler = false;
//         }
//         if(__glibc_likely(comp[i].status == IAX_COMP_SUCCESS)){
//           preempt_signal[0] = 0; /* reset to keep preemptible post-proc from yielding*/
//           fcontext_swap(child.prev_context, NULL);
//         }
//         else {
//           LOG_PRINT(LOG_ERR, "Offload failed: %d\n", comp[i].status);
//           return 0;
//         }

//         if(l_fidx == total_requests){
//           PRINT("Fillers completed\n");
//           break;
//         }
//       }
      break;
    case SERIAL_ACCESSOR:
    case SERIAL_ACCESSOR_DEP:
    case SERIAL_ACCESSOR_DECOMP:
    case SERIAL_ACCESSOR_DECOMP_DEP:
      for(i=0; i<total_requests; i++){
        state = (fcontext_state_t *)(&stack_ctxs[i * context_size]);
        fcontext_transfer_t child =  fcontext_swap(state->context, (void *)&dl_args[i]);
        preempt_signal = (uint8_t *)&comp[i];
        if(!new_filler){
          f_xfer = fcontext_swap(f_xfer.prev_context, NULL);
        } else {
new_filler_2:
          fstate = (fcontext_state_t *)(&f_stack_ctxs[l_fidx * context_size]);
          f_xfer = fcontext_swap(fstate->context, (void *)&f_args[l_fidx]);
        }
        if(f_args[l_fidx].done){
          new_filler = true;
          l_fidx++;
          if(comp[i].status == IAX_COMP_NONE){
            goto new_filler_2;
          }
        } else {
          new_filler = false;
        }
        if(__glibc_likely(comp[i].status == IAX_COMP_SUCCESS)){
          preempt_signal[0] = 0; /* reset to keep preemptible post-proc from yielding*/
          fcontext_swap(child.prev_context, NULL);
        }
        else {
          LOG_PRINT(LOG_ERR, "Offload failed: %d\n", comp[i].status);
          return 0;
        }

        if(l_fidx == total_requests){
          PRINT("Fillers completed\n");
          break;
        }
      }
      break;
    case DECRYPT_MEMCPY_DP:
//       for(i=0; i<total_requests; i++){
//         state = (fcontext_state_t *)(&stack_ctxs[i * context_size]);
//         fcontext_transfer_t child =  fcontext_swap(state->context, (void *)&dmdp_args[i]);
//         preempt_signal = (uint8_t *)&comp[i];
//         if(!new_filler){
//           f_xfer = fcontext_swap(f_xfer.prev_context, NULL);
//         } else {
// new_filler_3:
//           fstate = (fcontext_state_t *)(&f_stack_ctxs[l_fidx * context_size]);
//           f_xfer = fcontext_swap(fstate->context, (void *)&f_args[l_fidx]);
//         }
//         if(f_args[l_fidx].done){
//           new_filler = true;
//           l_fidx++;
//           if(comp[i].status == IAX_COMP_NONE){
//             goto new_filler_3;
//           }
//         } else {
//           new_filler = false;
//         }
//         if(__glibc_likely(comp[i].status == IAX_COMP_SUCCESS)){
//           preempt_signal[0] = 0; /* reset to keep preemptible post-proc from yielding*/
//           fcontext_swap(child.prev_context, NULL);
//         }
//         else {
//           LOG_PRINT(LOG_ERR, "Offload failed: %d\n", comp[i].status);
//           return 0;
//         }

//         if(l_fidx == total_requests){
//           PRINT("Fillers completed\n");
//           break;
//         }
//       }
      break;
    case MATMUL_MEMFILL_PCA:
//       for(i=0; i<total_requests; i++){
//         state = (fcontext_state_t *)(&stack_ctxs[i * context_size]);
//         fcontext_transfer_t child =  fcontext_swap(state->context, (void *)&mmpc_args[i]);
//         preempt_signal = (uint8_t *)&comp[i];
//         if(!new_filler){
//           f_xfer = fcontext_swap(f_xfer.prev_context, NULL);
//         } else {
// new_filler_4:
//           fstate = (fcontext_state_t *)(&f_stack_ctxs[l_fidx * context_size]);
//           f_xfer = fcontext_swap(fstate->context, (void *)&f_args[l_fidx]);
//         }
//         if(f_args[l_fidx].done){
//           new_filler = true;
//           l_fidx++;
//           if(comp[i].status == IAX_COMP_NONE){
//             goto new_filler_4;
//           }
//         } else {
//           new_filler = false;
//         }
//         if(__glibc_likely(comp[i].status == IAX_COMP_SUCCESS)){
//           preempt_signal[0] = 0; /* reset to keep preemptible post-proc from yielding*/
//           fcontext_swap(child.prev_context, NULL);
//           #ifdef SOJOURN
//           samples[i] = rdtsc();
//           #endif
//         }
//         else {
//           LOG_PRINT(LOG_ERR, "Offload failed: %d\n", comp[i].status);
//           return 0;
//         }

//         if(l_fidx == total_requests){
//           PRINT("Fillers completed\n");
//           break;
//         }
//       }
      break;
    case UPDATE_FILTER_HISTOGRAM:
//       for(i=0; i<total_requests; i++){
//         state = (fcontext_state_t *)(&stack_ctxs[i * context_size]);
//         fcontext_transfer_t child =  fcontext_swap(state->context, (void *)&upd_args[i]);
//         preempt_signal = (uint8_t *)&comp[i];
//         if(!new_filler){
//           f_xfer = fcontext_swap(f_xfer.prev_context, NULL);
//         } else {
// new_filler_5:
//           fstate = (fcontext_state_t *)(&f_stack_ctxs[l_fidx * context_size]);
//           f_xfer = fcontext_swap(fstate->context, (void *)&f_args[l_fidx]);
//         }
//         if(f_args[l_fidx].done){
//           new_filler = true;
//           l_fidx++;
//           if(comp[i].status == IAX_COMP_NONE){
//             goto new_filler_5;
//           }
//         } else {
//           new_filler = false;
//         }
//         if(__glibc_likely(comp[i].status == IAX_COMP_SUCCESS)){
//           preempt_signal[0] = 0; /* reset to keep preemptible post-proc from yielding*/
//           fcontext_swap(child.prev_context, NULL);
//         }
//         else {
//           LOG_PRINT(LOG_ERR, "Offload failed: %d\n", comp[i].status);
//           return 0;
//         }

//         if(l_fidx == total_requests){
//           PRINT("Fillers completed\n");
//           break;
//         }
//       }
      break;
    default:
      break;
  }

  #ifdef THROUGHPUT
    end = rdtsc();
    st_times[k] = start;
    end_times[k] = end;
  }
  avg_samples_from_arrays(exe_time_diffs, exe_times_avg, end_times, st_times, iter);
  PRINT("Yield&Preempt-Filler ");
  PRINT(" %lu", payload_size);
  mean_median_stdev_rps(exe_time_diffs, iter, total_requests, " RPS");

  #endif

	#ifdef EXETIME
  PRINT("Yield&&PreemptFillerFunction");
  PRINT(" %lu", payload_size);
  avg_samples_from_arrays(diff, pre_proc_times, ts1, ts0, total_requests);
  PRINT( " %lu", pre_proc_times);

  avg_samples_from_arrays(diff, offload_tax_times, ts2, ts1, total_requests);
  PRINT( " %lu", offload_tax_times);

  avg_samples_from_arrays(diff, ax_func_times, ts3, ts2, total_requests);
  PRINT( " %lu", ax_func_times);

  avg_samples_from_arrays(diff, post_proc_times, ts4, ts3, total_requests);
  PRINT( " %lu\n", post_proc_times);

  PRINT( "FillersCompleted: %d\n", l_fidx);
  #endif

  #ifdef THROUGHPUT
  #endif

  #ifdef SOJOURN
  }
  #endif
  #if defined(SOJOURN) || defined(THROUGHPUT) || defined(EXETIME)

yp_same:

  #if defined (SOJOURN) || defined(THROUGHPUT)
  for(int k=0; k<iter; k++){
  #endif


    dummy = (struct completion_record *)malloc(sizeof(struct completion_record));
    dummy->status = IAX_COMP_NONE;
    preempt_signal = (uint8_t *)dummy; /* noone is preempted */

    /* get arg length for generic executor */
    switch(main_type){
      case DESER_DECOMP_HASH:
        arg_len = sizeof(struct deser_decomp_hash_args);
        m_rq_fn = deser_decomp_hash_yielding;
        m_args = (char *)ddh_args;
        break;
      case DECRYPT_MEMCPY_DP:
        m_rq_fn = decrypt_memcpy_dotproduct_yielding;
        arg_len = sizeof(dmdp_args_t);
        m_args = (char *)dmdp_args;
        break;
      case MATMUL_MEMFILL_PCA:
        arg_len = sizeof(struct matmul_memfill_pca_args);
        m_rq_fn = matmul_memfill_pca_yielding;
        m_args = (char *)mmpc_args;
        break;
      case UPDATE_FILTER_HISTOGRAM:
        arg_len = sizeof(struct update_filter_histogram_args);
        m_rq_fn = update_filter_histogram_yielding;
        m_args = (char *)upd_args;
        break;
      case MEMCPY_GATHER:
        flush_range((void *)s_bufs, payload_size * total_requests);
        arg_len = sizeof(struct decomp_gather_args);
        m_rq_fn = memcpy_gather_yielding;
        m_args = (char *)dg_args;
        break;
      case DECOMP_GATHER:
        flush_range((void *)s_bufs, max_comp_size * total_requests);
        arg_len = sizeof(struct decomp_gather_args);
        m_rq_fn = decomp_gather_yielding;
        m_args = (char *)dg_args;
        break;
      default:
        PRINT("Unsupported main_type\n");
        return -1;
    }

    offload_req_xfer = (fcontext_transfer_t *)malloc(sizeof(fcontext_transfer_t) * total_requests);
    off_req_state = (fcontext_state_t **)malloc(sizeof(fcontext_state_t *) * total_requests);
    for(int i=0; i<total_requests; i++){
      off_req_state[i] = fcontext_create(m_rq_fn);
    }

    memset(comp, 0, total_requests * sizeof(idxd_comp));
    memset(desc, 0, total_requests * sizeof(idxd_desc));

    preempt_signal = (uint8_t *)&comp[nrq2c];
    ret_args.status = REQUEST_STATUS_NONE;

    nrq2c = 0;
    nurq = 0;
    prqidx = -1;

    #if defined(SOJOURN) || defined(THROUGHPUT)
    start = rdtsc();
    #endif

    while(nrq2c < total_requests){
      if(comp[nrq2c].status == IAX_COMP_SUCCESS){
        comp[nrq2c].status = IAX_COMP_NONE;
        LOG_PRINT(LOG_DEBUG, "OffloadCompResume Request %d\n", nrq2c);
        offload_req_xfer[nrq2c] =
          fcontext_swap(offload_req_xfer[nrq2c].prev_context, NULL);
        #ifdef SOJOURN
        samples[nrq2c] = rdtsc();
        #endif
        nrq2c++;
        if(__glibc_likely(nrq2c < total_requests)){
          preempt_signal = (uint8_t *)&comp[nrq2c];
        }
      }
      else if(prqidx > -1){
        LOG_PRINT(LOG_DEBUG, "PremptResume Request %d\n", prqidx);
        offload_req_xfer[prqidx] =
          fcontext_swap(offload_req_xfer[prqidx].prev_context, NULL);
        if(ret_args.status == REQUEST_COMPLETED ||
           ret_args.status == REQUEST_YIELDED){
          prqidx = -1;
        }
      }
      else if(nurq < total_requests){
        state =
          (fcontext_state_t *)(&stack_ctxs[nurq * context_size]);
        LOG_PRINT(LOG_DEBUG, "Start Request %d\n", nurq);
        offload_req_xfer[nurq] =
          fcontext_swap(off_req_state[nurq]->context, (void *)(m_args + nurq * arg_len));
        if(ret_args.status == REQUEST_PREEMPTED){
          LOG_PRINT(LOG_DEBUG, "Request %d preempted\n", nurq);
          prqidx = nurq;
        }
        nurq++;
      }
    }

    #ifdef THROUGHPUT
    end = rdtsc();
    st_times[k] = start;
    end_times[k] = end;
    #endif

    for(int i=0; i<total_requests; i++){
      fcontext_destroy(off_req_state[i]);
    }

    free(off_req_state);
    free(offload_req_xfer);

    #ifdef SOJOURN
    for(int i=0; i<num_indices; i++){
      int j = indices[i];
      index_samples[i][k] = samples[j];
    }
    start_dup_array[k] = start;
    #endif

  #if defined(SOJOURN) || defined(THROUGHPUT)
  }
  #endif

  #ifdef SOJOURN
  PRINT("PayloadSize QueueIdx ModeOfExe SojournTime\n");
  for(int i=0; i<num_indices; i++){
    int j = indices[i];
    avg_samples_from_arrays(sojourn_times, avg, index_samples[i], start_dup_array, iter);
    PRINT("%lu %d %s %lu\n", payload_size, j, "Yield&Preempt", avg);
  }
  #endif

  #ifdef THROUGHPUT
  avg_samples_from_arrays(exe_time_diffs, exe_times_avg, end_times, st_times, iter);
  PRINT("Yield&Preempt ");
  PRINT(" %lu", payload_size);
  mean_median_stdev_rps(exe_time_diffs, iter, total_requests, " RPS");
  #endif

  #ifdef EXETIME

  PRINT("Yielding ");
  PRINT(" %lu", payload_size);
  avg_samples_from_arrays(diff, pre_proc_times, ts1, ts0, total_requests);
  PRINT( " %lu", pre_proc_times);

  avg_samples_from_arrays(diff, offload_tax_times, ts2, ts1, total_requests);
  PRINT( " %lu", offload_tax_times);

  avg_samples_from_arrays(diff, ax_func_times, ts3, ts2, total_requests);
  PRINT( " %lu", ax_func_times);

  avg_samples_from_arrays(diff, post_proc_times, ts4, ts3, total_requests);
  PRINT( " %lu\n", post_proc_times);
  // return 0;
  #endif

  #if defined(SOJOURN) || defined(THROUGHPUT)
  for(int k=0; k<iter; k++){
  #endif

  switch(main_type){
    case DESER_DECOMP_HASH:
      arg_len = sizeof(struct deser_decomp_hash_args);
      m_rq_fn = deser_decomp_hash_yielding;
      m_args = (char *)ddh_args;
      break;
    case DECRYPT_MEMCPY_DP:
      m_rq_fn = decrypt_memcpy_dotproduct_yielding;
      arg_len = sizeof(dmdp_args_t);
      m_args = (char *)dmdp_args;
      break;

    case MATMUL_MEMFILL_PCA:
      arg_len = sizeof(struct matmul_memfill_pca_args);
      m_rq_fn = matmul_memfill_pca_yielding;
      m_args = (char *)mmpc_args;
      break;
    case UPDATE_FILTER_HISTOGRAM:
      arg_len = sizeof(struct update_filter_histogram_args);
      m_rq_fn = update_filter_histogram_yielding;
      m_args = (char *)upd_args;
      break;
    case MEMCPY_GATHER:
      flush_range((void *)s_bufs, payload_size * total_requests);
      arg_len = sizeof(dg_args_t);
      m_rq_fn = memcpy_gather_yielding;
      m_args = (char *)dg_args;
      break;
    case DECOMP_GATHER:
      flush_range((void *)s_bufs, max_comp_size * total_requests);
      arg_len = sizeof(dg_args_t);
      m_rq_fn = decomp_gather_yielding;
      m_args = (char *)dg_args;
      break;
    default:
      PRINT("Unsupported main_type\n");
      return -1;
  }
  dummy = (struct completion_record *)malloc(sizeof(struct completion_record));

  offload_req_xfer = (fcontext_transfer_t *)malloc(sizeof(fcontext_transfer_t) * total_requests);
  off_req_state = (fcontext_state_t **)malloc(sizeof(fcontext_state_t *) * total_requests);
  for(int i=0; i<total_requests; i++){
    off_req_state[i] = fcontext_create(m_rq_fn);
  }

  dummy->status = IAX_COMP_NONE;
  preempt_signal = (uint8_t *)dummy; /* noone is preempted */

  nurq = 0;
  nrq2c = 0;

    memset(comp, 0, total_requests * sizeof(idxd_comp));
    memset(desc, 0, total_requests * sizeof(idxd_desc));

    #if defined(SOJOURN) || defined(THROUGHPUT)
    start = rdtsc();
    #endif

    while(nrq2c < total_requests){
      if(comp[nrq2c].status == IAX_COMP_SUCCESS){
        fcontext_swap(offload_req_xfer[nrq2c].prev_context, NULL);
        #ifdef SOJOURN
        samples[nrq2c] = rdtsc();
        #endif
        nrq2c++;
      }
      else if(nurq < total_requests){
        offload_req_xfer[nurq] =
          // fcontext_swap(off_req_state[nurq]->context, (void *)&mmpc_args[nurq]);
          fcontext_swap(off_req_state[nurq]->context, (void *)(m_args + nurq * arg_len));
        nurq++;
      }
    }

    #ifdef THROUGHPUT
    end = rdtsc();
    st_times[k] = start;
    end_times[k] = end;
    #endif

    for(int i=0; i<total_requests; i++){
      fcontext_destroy(off_req_state[i]);
    }

    free(off_req_state);
    free(offload_req_xfer);

    #ifdef SOJOURN
    for(int i=0; i<num_indices; i++){
      int j = indices[i];
      index_samples[i][k] = samples[j];
    }
    start_dup_array[k] = start;
    #endif

  #if defined(SOJOURN) || defined(THROUGHPUT)
  }
  #endif

  #ifdef SOJOURN
  PRINT("PayloadSize QueueIdx ModeOfExe SojournTime\n");
  for(int i=0; i<num_indices; i++){
    int j = indices[i];
    avg_samples_from_arrays(sojourn_times, avg, index_samples[i], start_dup_array, iter);
    PRINT("%lu %d %s %lu\n", payload_size, j, "Yield", avg);
  }
  #endif

  #ifdef THROUGHPUT
  avg_samples_from_arrays(exe_time_diffs, exe_times_avg, end_times, st_times, iter);
  PRINT("Yield ");
  PRINT(" %lu", payload_size);
  mean_median_stdev_rps(exe_time_diffs, iter, total_requests, " RPS");

  #endif

  #ifdef EXETIME
  PRINT("Yield&Preempt ");
  PRINT(" %lu", payload_size);
  avg_samples_from_arrays(diff, pre_proc_times, ts1, ts0, total_requests);
  PRINT( " %lu", pre_proc_times);

  avg_samples_from_arrays(diff, offload_tax_times, ts2, ts1, total_requests);
  PRINT( " %lu", offload_tax_times);

  avg_samples_from_arrays(diff, ax_func_times, ts3, ts2, total_requests);
  PRINT( " %lu", ax_func_times);

  avg_samples_from_arrays(diff, post_proc_times, ts4, ts3, total_requests);
  PRINT( " %lu\n", post_proc_times);
  // return 0;
  #endif

  #endif // THROUGHPUT || SOJOURN || EXETIME



  munmap(nm->base_addr, nm->size);
  free(nm);

  free_iaa_wq();
  free_dsa_wq();
  // fcontext_destroy_proxy(self);
}