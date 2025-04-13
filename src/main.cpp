#include <boost/lockfree/queue.hpp>
#include <boost/coroutine2/all.hpp>
#include <boost/bind/bind.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <atomic>
#include <pthread.h>
#include <algorithm>
#include <queue>
#include <random>

#include "thread_utils.h"
#include "timer_utils.h"
#include "print_utils.h"
#include "readerwritercircularbuffer.h"
#include "readerwriterqueue.h"
#include "numa_mem.h"

#include "iaa_offloads.h"
#include "dsa_offloads.h"

#include "src/protobuf/generated/src/protobuf/router.pb.h"
#include "gpcore_compress.h"
#include "lzdatagen.h"
#include "ch3_hash.h"

#include "decrypt.h"
#include "dotproduct.h"

#include "pointer_chase.h"
#include "payload_gen.h"

#include "gather.h"
#include "matmul_histogram.h"

#define DSA_START 4
#define IAA_START 5
#define IDXD_DEV_STEP 2
#define DISPATCH_RING_SIZE 4096
#define REQ_SIZE 128
#define INQ_SIZE 4096
#define MAX_SER_OVERHEAD_BYTES 128
#define STACK_SIZE (128 * 1024)

static constexpr uint64_t PAGE_SIZE = 1 << 30;

using namespace moodycamel;
using namespace boost::placeholders;

typedef boost::coroutines2::coroutine<void *> coro_t;

__thread coro_t::push_type *curr_yield;

#ifdef DEBUG
std::atomic<int> num_completed;
#endif

#ifdef LATENCY
std::atomic<uint64_t> workload_start_cycle;
#endif
std::atomic<uint64_t> num_requests = 0;
typedef struct completion_record idxd_comp;
typedef struct hw_desc idxd_desc;

typedef enum job_type {
  DESER_DECOMP_HASH,
  DECRYPT_MEMCPY_DP,
  DECOMP_GATHER,
  MEMCPY_GATHER,
  UPDATE_FILTER_HISTOGRAM,
  MATMUL_MEMFILL_PCA,
  NUM_APPS
} job_type_t;
typedef enum status {
  INIT,
  OFFLOADED,
  OFFLOAD_STARTED,
  PREEMPTED,
  COMPLETED
} status;
typedef enum dist {
  EXPONENTIAL,
  BIMODAL,
  EXTREME_BIMODAL,
  DETERMINISTIC
} dist_t;
typedef enum worker_type_ {
  WORKER_MS,
  WORKER_RR,
  WORKER_MS_CL_OPT,
  WORKER_NOACC,
  WORKER_BLOCKING,
  WORKER_RR_SW_FALLBACK
} worker_type_t;
typedef struct job_info {
  job_type_t jtype;
  status s;
  void *args;
  idxd_comp *comp;
  uint64_t failed_enq;
} job_info_t;

#pragma pack(push, 1)
struct req_hdr {
  uint64_t id;
  job_type req_type;
  void *w_args;
  #ifdef LATENCY
  bool tagged;
  uint64_t arrival;
  uint64_t completed;
  int num_services;
  #if defined(LOST_ENQ_TIME) || defined(COUNT_LOST_ENQ)
  uint64_t failed_enqs;
  #endif
  uint64_t injected;
  uint64_t dispatched;
  uint64_t first_served;
  uint64_t prefn_completed;
  uint64_t resumed1;
  uint64_t resumed2;
  uint64_t postfn_completed;

  uint64_t unloaded; // it will be cached in the workload gen here
  #endif

  char padding[REQ_SIZE - sizeof(uint64_t) - sizeof(job_type) - sizeof(void*)
  #ifdef LATENCY
  - sizeof(bool) - 10 * sizeof(uint64_t) - sizeof(int)
  #endif
  #if defined(LOST_ENQ_TIME) || defined(COUNT_LOST_ENQ)
  - sizeof(uint64_t)
  #endif
  ];
}__attribute__((aligned(REQ_SIZE)));
#pragma pack(pop)

typedef struct req_hdr hdr; // total no latency -- 128 bytes
typedef struct coro_info {
  coro_t::pull_type *coro;
  coro_t::push_type *yield;
  job_info_t *jinfo;
  int id;
  hdr h; // can we use a pointer?
} coro_info_t;

#ifdef JSQ
typedef struct worker_info {
  int dq_idx;
  int num_running_jobs;
  int dispatched_jobs;
  int *p_finished_jobs;
  friend bool operator< (worker_info const& lhs, worker_info const& rhs) {
    return lhs.num_running_jobs > rhs.num_running_jobs; // so that it's a min heap
  }
} worker_info_t;
#endif

typedef BlockingReaderWriterCircularBuffer<hdr> dispatch_ring_t;
typedef ReaderWriterQueue<hdr> wrkload_ring_t;
typedef BlockingReaderWriterQueue<hdr> response_ring_t;

typedef struct wrkld_gen_args_t_ {
  wrkload_ring_t *q;
  int num_reqs;
  int *pushed;
  uint64_t *start;
  uint64_t *end;
  uint64_t num_ddh_reqs;
  uint64_t num_dmdp_reqs;
  uint64_t num_dg_reqs;
  uint64_t num_mg_reqs;
  uint64_t num_ufh_reqs;
  uint64_t num_mmp_reqs;
  dist_t dist_type;
  double peak;
  double peak2;
  bool unloaded;
  char *stack_pool;
  #ifdef POISSON
  double load;
  #endif
  int server_node;
  pthread_barrier_t *start_barrier;
  pthread_barrier_t *exit_barrier;
} wrkld_gen_args_t;

typedef struct dispatch_args_t_ {
  dispatch_ring_t **wqs;
  wrkload_ring_t *inq;
  int num_requests;
  int num_workers;
  pthread_barrier_t *start_barrier;
  pthread_barrier_t *exit_barrier;
  #ifdef JSQ
  int **pp_finished_jobs;
  #endif
} dispatch_args_t;

typedef struct wrk_args_t_ {
  int id;
  int num_requests;
  int num_coros;
  int node;
  dispatch_ring_t *q;
  response_ring_t *c_queue;
  int dsa_dev_id;
  int iaa_dev_id;
  pthread_barrier_t *start_barrier;
  pthread_barrier_t *exit_barrier;
  char *stack_pool;
  job_info_t *jinfos;
  idxd_comp *cr_pool;
  #ifdef JSQ
  int *p_finished_jobs;
  #endif
} wrk_args_t;

typedef struct monitor_args {
  response_ring_t **c_queues;
  pthread_barrier_t *start_barrier;
  pthread_barrier_t *exit_barrier;
  int num_requests;
  int num_workers;
  int id;
  int coros_per_worker;
  int num_iaas;
  int num_dsas;
  int inqueue_size_elems;
  int dispatch_queue_num_elems;
  job_info_t *jinfos;
} monitor_args_t;

typedef struct deser_decomp_hash_args {
  idxd_desc *desc;
  idxd_comp *comp;
  char *s_buf;
  char *d_buf;
  int s_sz;
  uint32_t hash;
  int d_sz;
  int id;
  int p_off;
  #ifdef EXETIME
  uint64_t *ts0;
  uint64_t *ts1;
  uint64_t *ts2;
  uint64_t *ts3;
  uint64_t *ts4;
  #endif
} ddh_args_t;

typedef struct decrypt_memcpy_dp_args {
  idxd_desc *desc;
  idxd_comp *comp;
  char *enc_buf;
  char *dec_buf;
  char *dst_buf;
  float score;
  int sz;
  int id;
  int p_off;
  #ifdef EXETIME
  uint64_t *ts0;
  uint64_t *ts1;
  uint64_t *ts2;
  uint64_t *ts3;
  uint64_t *ts4;
  #endif
} dmdp_args_t;

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
  int id;
  #ifdef EXETIME
  uint64_t *ts0;
  uint64_t *ts1;
  uint64_t *ts2;
  uint64_t *ts3;
  uint64_t *ts4;
  #endif
} dg_args_t;

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

  int p_off;
  int id;
  uint64_t *ts0;
  uint64_t *ts1;
  uint64_t *ts2;
  uint64_t *ts3;
  uint64_t *ts4;
} ufh_args_t;

typedef struct matmul_memfill_pca_args {
  idxd_desc *desc;
  idxd_comp *comp;
  mm_data_t *mm_data;
  int *mean_vec;
  int mat_size_bytes;

  int p_off;
  int id;
  uint64_t *ts0;
  uint64_t *ts1;
  uint64_t *ts2;
  uint64_t *ts3;
  uint64_t *ts4;
} mmpc_args_t;

__thread struct acctest_context *m_iaa = NULL;
__thread struct acctest_context *m_dsa = NULL;
__thread IppsAES_GCMState *pState = NULL;

class SimpleStack {
private:
    std::size_t     size_;
    char *stack_pool_;

public:
    SimpleStack( char *stack_pool, std::size_t size = STACK_SIZE ) BOOST_NOEXCEPT_OR_NOTHROW :
        size_( size), stack_pool_(stack_pool) {
    }

    boost::context::stack_context allocate() {
        boost::context::stack_context sctx;
        sctx.size = size_;
        sctx.sp = stack_pool_ + sctx.size;
        return sctx;
    }
    void deallocate( boost::context::stack_context & sctx) BOOST_NOEXCEPT_OR_NOTHROW {
      BOOST_ASSERT( sctx.sp);
      // don't need to do anything, main is gonna free it
      void * vp = static_cast< char * >( sctx.sp) - sctx.size;
      free(vp);
    }
};


static __always_inline uint64_t rdtsc(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc": "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

static __always_inline uint64_t
get_monotonic_timestamp() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static __always_inline uint64_t
fenced_rdtscp(void)
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

int initialize_wq(struct acctest_context **dev, int dev_id, int wq_id, int wq_type){
  int tflags = TEST_FLAGS_BOF;
  int rc;

  *dev = acctest_init(tflags);
  rc = acctest_alloc(*dev, wq_type, dev_id, wq_id);
  if(rc != ACCTEST_STATUS_OK){
    LOG_PRINT(LOG_ERR, "Error allocating work queue: %d\n", rc);
    return rc;
  }
  return 0;
}
void thread_wq_alloc(struct acctest_context **m_dev, int dev_id){
  int tflags = TEST_FLAGS_BOF;
  int rc;
  int wq_type = SHARED;
  int wq_id = 0;

  for (;;){
    rc = initialize_wq(m_dev, dev_id, wq_id, wq_type);
    if(rc == 0){
      break;
    }
    LOG_PRINT(LOG_DEBUG, "Error allocating DSA work queue... retrying \n");
  }

}


void gen_ser_buf(int avail_out, char *p_val, char *dst, int *msg_size, int insize){
  router::RouterRequest req;

  req.set_key("/region/cluster/foo:key|#|etc"); // key is 32B string, value gets bigger up to 2MB
  req.set_operation(0);

  std::string valstring(p_val, insize);

  req.set_value(valstring);

  *msg_size = req.ByteSizeLong();
  if(*msg_size > avail_out){
    throw std::runtime_error("Not enough space to serialize");
    return;
  }
  req.SerializeToArray(dst, *msg_size);
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

#ifdef POISSON
void gen_interarrival_poisson(double lambda, uint64_t *offsets, int num_reqs, uint64_t cycles_per_sec, uint64_t seed=0){
  /* lambda specifies the number of requests per second */
  std::random_device rd;
  std::mt19937 gen(seed);
  std::exponential_distribution<> d(lambda);

  uint64_t cumulative_cycles = 0;
  for (int i = 0; i < num_reqs; ++i) {
      double interarrival_time = d(gen);
      cumulative_cycles += static_cast<uint64_t>(interarrival_time * cycles_per_sec);
      offsets[i] = cumulative_cycles;
  }

}
#endif

static inline void decrypt_feature(void *cipher_inp, void *plain_out, int input_size){
  Ipp8u *pKey = (Ipp8u *)"0123456789abcdef";
  Ipp8u *pIV = (Ipp8u *)"0123456789ab";
  int keysize = 16;
  int ivsize = 12;
  int aadSize = 16;
  Ipp8u aad[aadSize];
  IppStatus status;

  if(pState == NULL){
    int ippAES_GCM_ctx_size;
    status = ippsAES_GCMGetSize(&ippAES_GCM_ctx_size);
    if(status != ippStsNoErr){
      LOG_PRINT(LOG_ERR, "Failed to get AES GCM size\n");
    }
    pState = (IppsAES_GCMState *)malloc(ippAES_GCM_ctx_size);
    status = ippsAES_GCMInit(pKey, keysize, pState, ippAES_GCM_ctx_size);
    if(status != ippStsNoErr){
      LOG_PRINT(LOG_ERR, "Failed to init AES GCM\n");
    }
  }

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

void dmdp_noacc(job_info_t *jinfo, coro_t::push_type & yield){
  dmdp_args_t *args = static_cast<dmdp_args_t*>(jinfo->args);
  char *enc_buf = args->enc_buf;
  char *dec_buf = args->dec_buf;
  char *dst_buf = args->dst_buf;
  int sz = args->sz;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  int p_off = args->p_off;
  int out_sz;
  int id = args->id;
#ifdef EXETIME
  ts0 = rdtsc();
#endif

  decrypt_feature(enc_buf, dec_buf, sz);

#ifdef EXETIME
  ts1 = rdtsc();
  ts2 = rdtsc();
#endif

  memcpy(dst_buf, dec_buf, sz);

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  dotproduct(dst_buf, (void *)&(args->score), sz, &out_sz);

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
#ifdef DEBUG
  num_completed++;
#endif
}

void dmdp_yielding(job_info_t *jinfo, coro_t::push_type & yield){
  dmdp_args_t *args = static_cast<dmdp_args_t*>(jinfo->args);
  char *enc_buf = args->enc_buf;
  char *dec_buf = args->dec_buf;
  char *dst_buf = args->dst_buf;
  int sz = args->sz;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = jinfo->comp;
  int p_off = args->p_off;
  int out_sz;
  int id = args->id;
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

  if(enqcmd((void *)((char *)(m_dsa->wq_reg) + p_off), desc)){
    memcpy(dst_buf, dec_buf, sz);
    comp->status = IAX_COMP_SUCCESS;
  } else {

#ifdef EXETIME
  ts2 = rdtsc();
#endif
  jinfo->s = OFFLOAD_STARTED;
  while(comp->status == IAX_COMP_NONE){
    jinfo->s = OFFLOADED;
    _mm_sfence();
    yield(&yield);
  }
  if(comp->status != IAX_COMP_SUCCESS && comp->status != IAX_COMP_NONE){
    PRINT( "Memcpy for request:%d failed: %d, size: %d\n", id, comp->status, sz);
    exit(1);
  }

  }

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  dotproduct(dst_buf, (void *)&(args->score), sz, &out_sz);

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

#ifdef DEBUG
  num_completed++;
#endif
}

void dmdp_yielding_rr_sw_fallback(job_info_t *jinfo, coro_t::push_type & yield){
  dmdp_args_t *args = static_cast<dmdp_args_t*>(jinfo->args);
  char *enc_buf = args->enc_buf;
  char *dec_buf = args->dec_buf;
  char *dst_buf = args->dst_buf;
  int sz = args->sz;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  int p_off = args->p_off;
  int out_sz;
  int id = args->id;
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

  if(enqcmd((void *)((char *)(m_dsa->wq_reg) + p_off), desc)){
    memcpy(dst_buf, dec_buf, sz);
    comp->status = IAX_COMP_SUCCESS;
  } else {

#ifdef EXETIME
  ts2 = rdtsc();
#endif
  jinfo->s = OFFLOAD_STARTED;
  while(comp->status == IAX_COMP_NONE){
    jinfo->s = OFFLOADED;
    _mm_sfence();
    yield(&yield);
  }
  if(comp->status != IAX_COMP_SUCCESS && comp->status != IAX_COMP_NONE){
    PRINT( "Memcpy for request:%d failed: %d, size: %d\n", id, comp->status, sz);
    exit(1);
  }

  }

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  dotproduct(dst_buf, (void *)&(args->score), sz, &out_sz);

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

#ifdef DEBUG
  num_completed++;
#endif
}

static __always_inline void dmdp_yielding_rr(job_info_t *jinfo, coro_t::push_type & yield){
  dmdp_args_t *args = static_cast<dmdp_args_t*>(jinfo->args);
  char *enc_buf = args->enc_buf;
  char *dec_buf = args->dec_buf;
  char *dst_buf = args->dst_buf;
  int sz = args->sz;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  int p_off = args->p_off;
  int out_sz;
  int id = args->id;
  #ifdef LOST_ENQ_TIME
  uint64_t start;
  #endif

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

  #ifdef LOST_ENQ_TIME
  start = fenced_rdtscp();
  #endif

  while(enqcmd((void *)((char *)(m_dsa->wq_reg) + args->p_off), desc) ){
    #ifdef COUNT_FAILED_ENQS
    jinfo->failed_enq++;
    #endif
  }

  #ifdef LOST_ENQ_TIME
  jinfo->failed_enq += (fenced_rdtscp() - start);
  #endif

#ifdef EXETIME
  ts2 = rdtsc();
#endif
  jinfo->s = OFFLOAD_STARTED;
  while(comp->status == IAX_COMP_NONE){
    jinfo->s = OFFLOADED;
    _mm_sfence();
    yield(&yield);
  }
  if(comp->status != IAX_COMP_SUCCESS && comp->status != IAX_COMP_NONE){
    PRINT("Memcpy for request:%d failed: %d, size: %d\n", id, comp->status, sz);
    exit(1);
  }

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  dotproduct(dst_buf, (void *)&(args->score), sz, &out_sz);

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

#ifdef DEBUG
  num_completed++;
#endif
}

static __always_inline void dmdp_blocking(job_info_t *jinfo, coro_t::push_type & yield){
  dmdp_args_t *args = static_cast<dmdp_args_t*>(jinfo->args);
  char *enc_buf = args->enc_buf;
  char *dec_buf = args->dec_buf;
  char *dst_buf = args->dst_buf;
  int sz = args->sz;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  int p_off = args->p_off;
  int out_sz;
  int id = args->id;
  #ifdef LOST_ENQ_TIME
  uint64_t start;
  #endif

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

  #ifdef LOST_ENQ_TIME
  start = fenced_rdtscp();
  #endif

  while(enqcmd((void *)((char *)(m_dsa->wq_reg) + args->p_off), desc) ){}

  #ifdef LOST_ENQ_TIME
  jinfo->failed_enq += (fenced_rdtscp() - start);
  #endif

#ifdef EXETIME
  ts2 = rdtsc();
#endif
  jinfo->s = OFFLOADED;
  while(comp->status == IAX_COMP_NONE){  }

#ifdef EXETIME
  ts3 = rdtsc();
#endif

  dotproduct(dst_buf, (void *)&(args->score), sz, &out_sz);

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

#ifdef DEBUG
  num_completed++;
#endif
}

static __always_inline void ddh_yielding(job_info_t *jinfo, coro_t::push_type & yield){
  ddh_args_t *args = static_cast<ddh_args_t*>(jinfo->args);
  router::RouterRequest req;
  int p_off = args->p_off;
  int s_sz = args->s_sz;
  char *s_buf = (char *)args->s_buf;
  char *d_buf = (char *)args->d_buf;
  int d_sz = args->d_sz;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = jinfo->comp;
  int id = args->id;
  unsigned long d_out_spc = IAA_DECOMPRESS_MAX_DEST_SIZE;
  #ifdef LOST_ENQ_TIME
  uint64_t start;
  #endif
  #ifdef EXETIME
  ts0 = rdtsc();
  #endif

  req.ParseFromArray(s_buf, s_sz);
#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_iaa_decompress_desc_with_preallocated_comp(
    desc, (uint64_t)(&(req.value())[0]), (uint64_t)(d_buf),
    (uint64_t)(comp), req.value().size());

  #ifdef LOST_ENQ_TIME
  start = fenced_rdtscp();
  #endif

  if(enqcmd((void *)((char *)(m_iaa->wq_reg) + p_off), desc)){
    int rc =
      gpcore_do_decompress((void *)d_buf, (void *)(&(req.value())[0]), req.value().size(),
        &d_out_spc);
    if(rc != 0){
      PRINT( "Decompression for request:%d failed: %d\n", id, rc);
      exit(1);
    }
    comp->status = IAX_COMP_SUCCESS;
  } else {


    #ifdef COUNT_FAILED_ENQS
    jinfo->failed_enq++;
    #endif

    #ifdef LOST_ENQ_TIME
    jinfo->failed_enq += (fenced_rdtscp() - start);
    #endif

    jinfo->s = OFFLOAD_STARTED;
    while(comp->status == IAX_COMP_NONE){
      jinfo->s = OFFLOADED;
      _mm_sfence();
      yield(&yield);
    }
    if(comp->status != IAX_COMP_SUCCESS && comp->status != IAX_COMP_NONE){
      PRINT( "Decompression for request:%d failed: %d\n", id, comp->status);
      exit(1);
    }

  }

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

#ifdef DEBUG
  num_completed++;
#endif
}

static __always_inline void ddh_yielding_rr(job_info_t *jinfo, coro_t::push_type & yield){
  ddh_args_t *args = static_cast<ddh_args_t*>(jinfo->args);
  router::RouterRequest req;
  int p_off = args->p_off;
  int s_sz = args->s_sz;
  char *s_buf = (char *)args->s_buf;
  char *d_buf = (char *)args->d_buf;
  int d_sz = args->d_sz;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  int id = args->id;
  unsigned long d_out_spc = IAA_DECOMPRESS_MAX_DEST_SIZE;
  #ifdef LOST_ENQ_TIME
  uint64_t start;
  #endif
  #ifdef EXETIME
  ts0 = rdtsc();
  #endif

  req.ParseFromArray(s_buf, s_sz);
#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_iaa_decompress_desc_with_preallocated_comp(
    desc, (uint64_t)(&(req.value())[0]), (uint64_t)(d_buf),
    (uint64_t)(comp), req.value().size());

  #ifdef LOST_ENQ_TIME
  start = fenced_rdtscp();
  #endif
  while(enqcmd((void *)((char *)(m_iaa->wq_reg) + p_off), desc) ){
    /* retry submit */
    #ifdef COUNT_FAILED_ENQS
    jinfo->failed_enq++;
    #endif
  }
  #ifdef LOST_ENQ_TIME
  jinfo->failed_enq += (fenced_rdtscp() - start);
  #endif

#ifdef EXETIME
  ts2 = rdtsc();
#endif
  jinfo->s = OFFLOAD_STARTED;
  while(comp->status == IAX_COMP_NONE){
    jinfo->s = OFFLOADED;
    _mm_sfence();
    yield(&yield);
  }
  if(comp->status != IAX_COMP_SUCCESS && comp->status != IAX_COMP_NONE){
    PRINT("Decompression for request:%d failed: %d\n", id, comp->status);
    exit(1);
  }
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

#ifdef DEBUG
  num_completed++;
#endif
}

static __always_inline void ddh_yielding_rr_sw_fallback(job_info_t *jinfo, coro_t::push_type & yield){
  ddh_args_t *args = static_cast<ddh_args_t*>(jinfo->args);
  router::RouterRequest req;
  int p_off = args->p_off;
  int s_sz = args->s_sz;
  char *s_buf = (char *)args->s_buf;
  char *d_buf = (char *)args->d_buf;
  int d_sz = args->d_sz;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  int id = args->id;
  unsigned long d_out_spc = IAA_DECOMPRESS_MAX_DEST_SIZE;
  #ifdef LOST_ENQ_TIME
  uint64_t start;
  #endif
  #ifdef EXETIME
  ts0 = rdtsc();
  #endif

  req.ParseFromArray(s_buf, s_sz);
#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_iaa_decompress_desc_with_preallocated_comp(
    desc, (uint64_t)(&(req.value())[0]), (uint64_t)(d_buf),
    (uint64_t)(comp), req.value().size());


  if(enqcmd((void *)((char *)(m_iaa->wq_reg) + p_off), desc)){
    int rc =
      gpcore_do_decompress((void *)d_buf, (void *)(&(req.value())[0]), req.value().size(),
        &d_out_spc);
    if(rc != 0){
      PRINT( "Decompression for request:%d failed: %d\n", id, rc);
      exit(1);
    }
    comp->status = IAX_COMP_SUCCESS;
  } else {

    #ifdef LOST_ENQ_TIME
  start = fenced_rdtscp();
    #endif
    #ifdef COUNT_FAILED_ENQS
    jinfo->failed_enq++;
    #endif

    #ifdef LOST_ENQ_TIME
  jinfo->failed_enq += (fenced_rdtscp() - start);
    #endif

  jinfo->s = OFFLOAD_STARTED;
  while(comp->status == IAX_COMP_NONE){
    jinfo->s = OFFLOADED;
    _mm_sfence();
    yield(&yield);
  }
  if(comp->status != IAX_COMP_SUCCESS && comp->status != IAX_COMP_NONE){
    PRINT( "Decompression for request:%d failed: %d\n", id, comp->status);
    exit(1);
  }
  }



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

#ifdef DEBUG
  num_completed++;
#endif
}

static __always_inline void ddh_blocking(job_info_t *jinfo, coro_t::push_type & yield){
  ddh_args_t *args = static_cast<ddh_args_t*>(jinfo->args);
  router::RouterRequest req;
  int p_off = args->p_off;
  int s_sz = args->s_sz;
  char *s_buf = (char *)args->s_buf;
  char *d_buf = (char *)args->d_buf;
  int d_sz = args->d_sz;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  int id = args->id;
  unsigned long d_out_spc = IAA_DECOMPRESS_MAX_DEST_SIZE;
  #ifdef LOST_ENQ_TIME
  uint64_t start;
  #endif
  #ifdef EXETIME
  ts0 = rdtsc();
  #endif

  req.ParseFromArray(s_buf, s_sz);
#ifdef EXETIME
  ts1 = rdtsc();
#endif

  prepare_iaa_decompress_desc_with_preallocated_comp(
    desc, (uint64_t)(&(req.value())[0]), (uint64_t)(d_buf),
    (uint64_t)(comp), req.value().size());

  #ifdef LOST_ENQ_TIME
  start = fenced_rdtscp();
  #endif

  while(enqcmd((void *)((char *)(m_iaa->wq_reg) + p_off), desc) ){
    /* retry submit */
  }

  #ifdef LOST_ENQ_TIME
  jinfo->failed_enq += (fenced_rdtscp() - start);
  #endif

#ifdef EXETIME
  ts2 = rdtsc();
#endif
  jinfo->s = OFFLOADED;
  while(comp->status == IAX_COMP_NONE){
  }

  if(comp->status != IAX_COMP_SUCCESS && comp->status != IAX_COMP_NONE){
    PRINT("Decompression for request:%d failed: %d\n", id, comp->status);
    exit(1);
  }
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

#ifdef DEBUG
  num_completed++;
#endif
}

static __always_inline void ddh_noacc(job_info_t *jinfo, coro_t::push_type & yield){
  ddh_args_t *args = static_cast<ddh_args_t*>(jinfo->args);
  router::RouterRequest req;
  int p_off = args->p_off;
  int s_sz = args->s_sz;
  char *s_buf = (char *)args->s_buf;
  char *d_buf = (char *)args->d_buf;
  int d_sz = args->d_sz;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = args->comp;
  int id = args->id;
  uLong decompSize = IAA_DECOMPRESS_MAX_DEST_SIZE;

  unsigned long d_out_spc = IAA_DECOMPRESS_MAX_DEST_SIZE;
  #ifdef EXETIME
  ts0 = rdtsc();
  #endif

  req.ParseFromArray(s_buf, s_sz);
#ifdef EXETIME
  ts1 = rdtsc();
#endif


#ifdef EXETIME
  ts2 = rdtsc();
#endif
  int rc =
    gpcore_do_decompress((void *)d_buf, (void *)(&(req.value())[0]), req.value().size(),
      &decompSize);
  if(rc != 0){
    PRINT( "Decompression for request:%d failed: %d\n", id, rc);
    exit(1);
  }
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

#ifdef DEBUG
  num_completed++;
#endif
}

static __always_inline void decomp_gather_yielding(job_info_t *jinfo, coro_t::push_type & yield){
  struct decomp_gather_args *args = static_cast<struct decomp_gather_args*>(jinfo->args);

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
  idxd_comp *comp = jinfo->comp;
  uLong d_out_spc = IAA_DECOMPRESS_MAX_DEST_SIZE;

  prepare_iaa_decompress_desc_with_preallocated_comp(
    desc, (uint64_t)(s_buf), (uint64_t)d_buf,
    (uint64_t)comp, s_sz);

  if(enqcmd((void *)((char *)(m_iaa->wq_reg) + p_off), desc)){
    int rc =
      gpcore_do_decompress((void *)d_buf, (void *)s_buf, s_sz,
        &d_out_spc);
    if(rc != 0){
      PRINT( "Decompression for request failed: %d\n", rc);
      exit(1);
    }
    d_sz = d_out_spc;
    comp->status = IAX_COMP_SUCCESS;
  } else {

  jinfo->s = OFFLOAD_STARTED;
  while(comp->status == IAX_COMP_NONE){
    jinfo->s = OFFLOADED;
    _mm_sfence();
    yield(&yield);
  }
  LOG_PRINT(LOG_TOO_VERBOSE, "DecompData[0:5]: %f%f%f%f%f\n", d_buf[0], d_buf[1], d_buf[2], d_buf[3], d_buf[4]);
  LOG_PRINT(LOG_TOO_VERBOSE, "DecompData[Last5]: %f%f%f%f%f\n", d_buf[d_sz-5], d_buf[d_sz-4], d_buf[d_sz-3], d_buf[d_sz-2], d_buf[d_sz-1]);
  if(comp->status != IAX_COMP_SUCCESS && comp->status != IAX_COMP_NONE){
    PRINT("Decompression for request failed: %d\n", comp->status);
    exit(1);
  }

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

}


static __always_inline  void decomp_gather_yielding_rr(job_info_t *jinfo, coro_t::push_type & yield){
  struct decomp_gather_args *args = static_cast<struct decomp_gather_args*>(jinfo->args);

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

  prepare_iaa_decompress_desc_with_preallocated_comp(
    desc, (uint64_t)(s_buf), (uint64_t)d_buf,
    (uint64_t)comp, s_sz);

  while(enqcmd((void *)((char *)(m_iaa->wq_reg) + p_off), desc) ){
  }


  jinfo->s = OFFLOAD_STARTED;
  while(comp->status == IAX_COMP_NONE){
    jinfo->s = OFFLOADED;
    _mm_sfence();
    yield(&yield);
  }
  LOG_PRINT(LOG_TOO_VERBOSE, "DecompData[0:5]: %f%f%f%f%f\n", d_buf[0], d_buf[1], d_buf[2], d_buf[3], d_buf[4]);
  LOG_PRINT(LOG_TOO_VERBOSE, "DecompData[Last5]: %f%f%f%f%f\n", d_buf[d_sz-5], d_buf[d_sz-4], d_buf[d_sz-3], d_buf[d_sz-2], d_buf[d_sz-1]);
  if(comp->status != IAX_COMP_SUCCESS && comp->status != IAX_COMP_NONE){
    PRINT("Decompression for request failed: %d\n", comp->status);
    exit(1);
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

}

static __always_inline void decomp_gather_yielding_rr_sw_fallback(job_info_t *jinfo, coro_t::push_type & yield){
  struct decomp_gather_args *args = static_cast<struct decomp_gather_args*>(jinfo->args);

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
  uLong d_out_spc = IAA_DECOMPRESS_MAX_DEST_SIZE;

  prepare_iaa_decompress_desc_with_preallocated_comp(
    desc, (uint64_t)(s_buf), (uint64_t)d_buf,
    (uint64_t)comp, s_sz);

  if(enqcmd((void *)((char *)(m_iaa->wq_reg) + p_off), desc)){
    int rc =
      gpcore_do_decompress((void *)d_buf, (void *)s_buf, s_sz,
        &d_out_spc);
    if(rc != 0){
      PRINT( "Decompression for request failed: %d\n", rc);
      exit(1);
    }
    d_sz = d_out_spc;
    comp->status = IAX_COMP_SUCCESS;
  } else {

  jinfo->s = OFFLOAD_STARTED;
  while(comp->status == IAX_COMP_NONE){
    jinfo->s = OFFLOADED;
    _mm_sfence();
    yield(&yield);
  }
  LOG_PRINT(LOG_TOO_VERBOSE, "DecompData[0:5]: %f%f%f%f%f\n", d_buf[0], d_buf[1], d_buf[2], d_buf[3], d_buf[4]);
  LOG_PRINT(LOG_TOO_VERBOSE, "DecompData[Last5]: %f%f%f%f%f\n", d_buf[d_sz-5], d_buf[d_sz-4], d_buf[d_sz-3], d_buf[d_sz-2], d_buf[d_sz-1]);
  if(comp->status != IAX_COMP_SUCCESS && comp->status != IAX_COMP_NONE){
    PRINT( "Decompression for request failed: %d\n", comp->status);
    exit(1);
  }

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

}


static __always_inline  void decomp_gather_blocking(job_info_t *jinfo, coro_t::push_type & yield){
  struct decomp_gather_args *args = static_cast<struct decomp_gather_args*>(jinfo->args);

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

  prepare_iaa_decompress_desc_with_preallocated_comp(
    desc, (uint64_t)(s_buf), (uint64_t)d_buf,
    (uint64_t)comp, s_sz);

  while(enqcmd((void *)((char *)(m_iaa->wq_reg) + p_off), desc) ){
  }


  jinfo->s = OFFLOAD_STARTED;
  while(comp->status == IAX_COMP_NONE){
  }
  LOG_PRINT(LOG_TOO_VERBOSE, "DecompData[0:5]: %f%f%f%f%f\n", d_buf[0], d_buf[1], d_buf[2], d_buf[3], d_buf[4]);
  LOG_PRINT(LOG_TOO_VERBOSE, "DecompData[Last5]: %f%f%f%f%f\n", d_buf[d_sz-5], d_buf[d_sz-4], d_buf[d_sz-3], d_buf[d_sz-2], d_buf[d_sz-1]);
  if(comp->status != IAX_COMP_SUCCESS && comp->status != IAX_COMP_NONE){
    PRINT( "Decompression for request failed: %d\n", comp->status);
    exit(1);
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

}

void decomp_gather_noacc(job_info_t *jinfo, coro_t::push_type & yield){
  struct decomp_gather_args *args = static_cast<struct decomp_gather_args*>(jinfo->args);

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
  uLong uncomp_size = IAA_DECOMPRESS_MAX_DEST_SIZE;

  int rc =
    gpcore_do_decompress((void *)d_buf, (void *)s_buf, s_sz,
      &uncomp_size);
  if(rc != 0){
    PRINT("Decompression failed: %d\n", rc);
    exit(1);
  }

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

static __always_inline void ufh_noacc(job_info_t *jinfo, coro_t::push_type & yield){
  ufh_args_t *args = static_cast<ufh_args_t*>(jinfo->args);

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

}

static __always_inline void ufh_blocking(job_info_t *jinfo, coro_t::push_type & yield){
  ufh_args_t *args = static_cast<ufh_args_t*>(jinfo->args);


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

  while(enqcmd((void *)((char *)(m_iaa->wq_reg) + args->p_off), desc) ){
    /* retry submit */
  }

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  while(comp->status == IAX_COMP_NONE){  }
  if(comp->status != IAX_COMP_SUCCESS){
    PRINT( "Extract failed: %d\n", comp->status);
    exit(-1);
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

}

static __always_inline void ufh_yielding(job_info_t *jinfo, coro_t::push_type & yield){
  ufh_args_t *args = static_cast<ufh_args_t*>(jinfo->args);


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
  idxd_comp *comp = jinfo->comp;

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

  if(enqcmd((void *)((char *)(m_iaa->wq_reg) + args->p_off), desc)){
    gpcore_do_extract( (uint8_t *)upd_buf, extracted, low_val, high_val, aecs);
    comp->status = IAX_COMP_SUCCESS;
  } else {
#ifdef EXETIME
  ts2 = rdtsc();
#endif

  jinfo->s = OFFLOAD_STARTED;
  while(comp->status == IAX_COMP_NONE){
    jinfo->s = OFFLOADED;
    _mm_sfence();
    yield(&yield);
  }
  if(comp->status != IAX_COMP_SUCCESS){
    PRINT( "Extract failed: %d\n", comp->status);
    exit(-1);
  }

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

}

static __always_inline void ufh_yielding_rr(job_info_t *jinfo, coro_t::push_type & yield){
  ufh_args_t *args = static_cast<ufh_args_t*>(jinfo->args);


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

  while(enqcmd((void *)((char *)(m_iaa->wq_reg) + args->p_off), desc) ){
    /* retry submit */
  }

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  jinfo->s = OFFLOAD_STARTED;
  while(comp->status == IAX_COMP_NONE){
    jinfo->s = OFFLOADED;
    _mm_sfence();
    yield(&yield);
  }
  if(comp->status != IAX_COMP_SUCCESS){
    PRINT( "Extract failed: %d\n", comp->status);
    exit(-1);
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

}

static __always_inline void ufh_yielding_rr_sw_fallback(job_info_t *jinfo, coro_t::push_type & yield){
  ufh_args_t *args = static_cast<ufh_args_t*>(jinfo->args);


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

  if(enqcmd((void *)((char *)(m_iaa->wq_reg) + args->p_off), desc)){
    gpcore_do_extract( (uint8_t *)upd_buf, extracted, low_val, high_val, aecs);
    comp->status = IAX_COMP_SUCCESS;
  } else {
#ifdef EXETIME
  ts2 = rdtsc();
#endif

  jinfo->s = OFFLOAD_STARTED;
  while(comp->status == IAX_COMP_NONE){
    jinfo->s = OFFLOADED;
    _mm_sfence();
    yield(&yield);
  }
  if(comp->status != IAX_COMP_SUCCESS){
    PRINT( "Extract failed: %d\n", comp->status);
    exit(-1);
  }

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

}

static __always_inline void mmp_yielding(job_info_t *jinfo, coro_t::push_type & yield){
  mmpc_args_t *args = static_cast<mmpc_args_t*>(jinfo->args);

  mm_data_t *mm_data = args->mm_data;
  uint64_t ts0, ts1, ts2, ts3, ts4;
  idxd_desc *desc = args->desc;
  idxd_comp *comp = jinfo->comp;
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

  if(enqcmd((void *)((char *)(m_dsa->wq_reg) + args->p_off), desc)){
    memset_pattern((void *)mm_data->matrix_out, 0xdeadbeef, mat_size_bytes/2);
    comp->status = IAX_COMP_SUCCESS;
  } else {

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  jinfo->s = OFFLOAD_STARTED;
  while(comp->status == IAX_COMP_NONE){
    jinfo->s = OFFLOADED;
    _mm_sfence();
    yield(&yield);
  }
  if(comp->status != IAX_COMP_SUCCESS){
    PRINT("Memfill failed: %d\n", comp->status);
    exit(-1);
  }

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

}

static __always_inline void mmp_yielding_rr(job_info_t *jinfo, coro_t::push_type & yield){
  mmpc_args_t *args = static_cast<mmpc_args_t*>(jinfo->args);

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

  while(enqcmd((void *)((char *)(m_dsa->wq_reg) + args->p_off), desc) ){ }

#ifdef EXETIME
  ts2 = rdtsc();
#endif
  jinfo->s = OFFLOAD_STARTED;
  while(comp->status == IAX_COMP_NONE){
    jinfo->s = OFFLOADED;
    _mm_sfence();
    yield(&yield);
  }
  if(comp->status != IAX_COMP_SUCCESS){
    PRINT("Memfill failed: %d\n", comp->status);
    exit(-1);
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

}

static __always_inline void mmp_yielding_rr_sw_fallback(job_info_t *jinfo, coro_t::push_type & yield){
  mmpc_args_t *args = static_cast<mmpc_args_t*>(jinfo->args);

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

  if(enqcmd((void *)((char *)(m_dsa->wq_reg) + args->p_off), desc)){
    memset_pattern((void *)mm_data->matrix_out, 0xdeadbeef, mat_size_bytes/2);
    comp->status = IAX_COMP_SUCCESS;
  } else {

#ifdef EXETIME
  ts2 = rdtsc();
#endif

  jinfo->s = OFFLOAD_STARTED;
  while(comp->status == IAX_COMP_NONE){
    jinfo->s = OFFLOADED;
    _mm_sfence();
    yield(&yield);
  }
  if(comp->status != IAX_COMP_SUCCESS){
    PRINT("Memfill failed: %d\n", comp->status);
    exit(-1);
  }

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

}

static __always_inline void mmp_blocking(job_info_t *jinfo, coro_t::push_type & yield){
  mmpc_args_t *args = static_cast<mmpc_args_t*>(jinfo->args);

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

  while(enqcmd((void *)((char *)(m_dsa->wq_reg) + args->p_off), desc) ){ }

#ifdef EXETIME
  ts2 = rdtsc();
#endif
  while(comp->status == IAX_COMP_NONE){
  }
  if(comp->status != IAX_COMP_SUCCESS){
    PRINT("Memfill failed: %d\n", comp->status);
    exit(-1);
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

}

static __always_inline void mmp_noacc(job_info_t *jinfo, coro_t::push_type & yield){
  mmpc_args_t *args = static_cast<mmpc_args_t*>(jinfo->args);

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

}

static __always_inline void memcpy_gather_blocking(job_info_t *jinfo, coro_t::push_type & yield){
  struct decomp_gather_args *args = static_cast<struct decomp_gather_args*>(jinfo->args);

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
  #ifdef LOST_ENQ_TIME
  uint64_t start;
  #endif
  uint64_t spin_cycles = 4200, start_cycle;

  prepare_dsa_memcpy_desc_with_preallocated_comp(
    desc, (uint64_t)(s_buf), (uint64_t)d_buf,
    (uint64_t)comp, s_sz);

  #ifdef LOST_ENQ_TIME
  start = fenced_rdtscp();
  #endif

  while(enqcmd((void *)((char *)(m_dsa->wq_reg) + p_off), desc) ){
  }

  #ifdef LOST_ENQ_TIME
  jinfo->failed_enq = (fenced_rdtscp() - start);
  #endif

  jinfo->s = OFFLOAD_STARTED;
  while(comp->status == IAX_COMP_NONE){
  }
  if(comp->status != IAX_COMP_SUCCESS && comp->status != IAX_COMP_NONE){
    PRINT("Memcpy for request failed: %d\n", comp->status);
    exit(1);
  }
#ifdef EXETIME
  ts3 = rdtsc();
#endif

  /* spin for t microseconds */
  start_cycle = rdtsc();
  while(rdtsc() - start_cycle < spin_cycles){}

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

}

static __always_inline void memcpy_gather_yielding_rr(job_info_t *jinfo, coro_t::push_type & yield){
  struct decomp_gather_args *args = static_cast<struct decomp_gather_args*>(jinfo->args);

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
  #ifdef LOST_ENQ_TIME
  uint64_t start;
  #endif
  uint64_t spin_cycles = 4200, start_cycle;

  prepare_dsa_memcpy_desc_with_preallocated_comp(
    desc, (uint64_t)(s_buf), (uint64_t)d_buf,
    (uint64_t)comp, s_sz);

  #ifdef LOST_ENQ_TIME
  start = fenced_rdtscp();
  #endif

  #ifdef LOST_ENQ_TIME
  start = fenced_rdtscp();
  #endif

  while(enqcmd((void *)((char *)(m_dsa->wq_reg) + p_off), desc) ){
  }

  #ifdef LOST_ENQ_TIME
  jinfo->failed_enq = (fenced_rdtscp() - start);
  #endif

  #ifdef LOST_ENQ_TIME
  jinfo->failed_enq = (fenced_rdtscp() - start);
  #endif


  jinfo->s = OFFLOAD_STARTED;
  while(comp->status == IAX_COMP_NONE){
    jinfo->s = OFFLOADED;
    _mm_sfence();
    yield(&yield);
  }
  if(comp->status != IAX_COMP_SUCCESS && comp->status != IAX_COMP_NONE){
    LOG_PRINT(LOG_ERR, "Memcpy for request failed: %d\n", comp->status);
    exit(1);
  }
#ifdef EXETIME
  ts3 = rdtsc();
#endif

  /* spin for t microseconds */
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

}

static __always_inline void memcpy_gather_yielding_rr_sw_fallback(job_info_t *jinfo, coro_t::push_type & yield){
  struct decomp_gather_args *args = static_cast<struct decomp_gather_args*>(jinfo->args);

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

  prepare_dsa_memcpy_desc_with_preallocated_comp(
    desc, (uint64_t)(s_buf), (uint64_t)d_buf,
    (uint64_t)comp, s_sz);

  if(enqcmd((void *)((char *)(m_dsa->wq_reg) + p_off), desc)){
    memcpy(d_buf, s_buf, s_sz);
    comp->status = IAX_COMP_SUCCESS;
  } else {

  jinfo->s = OFFLOAD_STARTED;
  while(comp->status == IAX_COMP_NONE){
    jinfo->s = OFFLOADED;
    _mm_sfence();
    yield(&yield);
  }
  if(comp->status != IAX_COMP_SUCCESS && comp->status != IAX_COMP_NONE){
    LOG_PRINT(LOG_ERR, "Memcpy for request failed: %d\n", comp->status);
    exit(1);
  }

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

}

static __always_inline void memcpy_gather_yielding(job_info_t *jinfo, coro_t::push_type & yield){
  struct decomp_gather_args *args = static_cast<struct decomp_gather_args*>(jinfo->args);

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
  idxd_comp *comp = jinfo->comp;

  prepare_dsa_memcpy_desc_with_preallocated_comp(
    desc, (uint64_t)(s_buf), (uint64_t)d_buf,
    (uint64_t)comp, s_sz);

  if(enqcmd((void *)((char *)(m_dsa->wq_reg) + p_off), desc)){
    memcpy(d_buf, s_buf, s_sz);
    comp->status = IAX_COMP_SUCCESS;
  } else {

  jinfo->s = OFFLOAD_STARTED;
  while(comp->status == IAX_COMP_NONE){
    jinfo->s = OFFLOADED;
    _mm_sfence();
    yield(&yield);
  }
  if(comp->status != IAX_COMP_SUCCESS && comp->status != IAX_COMP_NONE){
    LOG_PRINT(LOG_ERR, "Memcpy for request failed: %d\n", comp->status);
    exit(1);
  }

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

}

void memcpy_gather_noacc(job_info_t *jinfo, coro_t::push_type & yield){
  struct decomp_gather_args *args = static_cast<struct decomp_gather_args*>(jinfo->args);

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

  uint64_t spin_cycles = 4200, start_cycle;

  memcpy(d_buf, s_buf, s_sz);
#ifdef EXETIME
  ts3 = rdtsc();
#endif

  /* spin for t microseconds */
  start_cycle = rdtsc();
  while(rdtsc() - start_cycle < spin_cycles){}


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

}


void coro_noacc(int id, job_info_t *&jinfo, coro_t::push_type &yield)
{
  LOG_PRINT(LOG_DEBUG, "[coro]: coro %d is ready!\n", id);
  yield(&yield);
  for(;;){
    switch(jinfo->jtype){
      case DESER_DECOMP_HASH:
        ddh_noacc(jinfo, yield);
        break;
      case DECRYPT_MEMCPY_DP:
        dmdp_noacc(jinfo, yield);
        break;
      case DECOMP_GATHER:
        decomp_gather_noacc(jinfo, yield);
        break;
      case MEMCPY_GATHER:
        memcpy_gather_noacc(jinfo, yield);
        break;
      case UPDATE_FILTER_HISTOGRAM:
        ufh_noacc(jinfo, yield);
        break;
      case MATMUL_MEMFILL_PCA:
        mmp_noacc(jinfo, yield);
        break;
      default:
        std::cout << "Unknown job type" << std::endl;
        break;
    }
    jinfo->s = COMPLETED;
    yield(&yield);
  }
}

void coro_blocking(int id, job_info_t *&jinfo, coro_t::push_type &yield)
{
  LOG_PRINT(LOG_DEBUG, "[coro]: coro %d is ready!\n", id);
  yield(&yield);
  for(;;){
    switch(jinfo->jtype){
      case DESER_DECOMP_HASH:
        ddh_blocking(jinfo, yield);
        break;
      case DECRYPT_MEMCPY_DP:
        dmdp_blocking(jinfo, yield);
        break;
      case DECOMP_GATHER:
        decomp_gather_blocking(jinfo, yield);
        break;
      case MEMCPY_GATHER:
        memcpy_gather_blocking(jinfo, yield);
        break;
      case UPDATE_FILTER_HISTOGRAM:
        ufh_blocking(jinfo, yield);
        break;
      case MATMUL_MEMFILL_PCA:
        mmp_blocking(jinfo, yield);
        break;
      default:
        std::cout << "Unknown job type" << std::endl;
        break;
    }
    jinfo->s = COMPLETED;
    yield(&yield);
  }
}

void coro(int id, job_info_t *&jinfo, coro_t::push_type &yield)
{
  LOG_PRINT(LOG_DEBUG, "[coro]: coro %d is ready!\n", id);
  yield(&yield);
  for(;;){
    switch(jinfo->jtype){
      case DESER_DECOMP_HASH:
        ddh_yielding(jinfo, yield);
        break;
      case DECRYPT_MEMCPY_DP:
        dmdp_yielding(jinfo, yield);
        break;
      case DECOMP_GATHER:
        decomp_gather_yielding(jinfo, yield);
        break;
      case MEMCPY_GATHER:
        memcpy_gather_yielding(jinfo, yield);
        break;
      case UPDATE_FILTER_HISTOGRAM:
        ufh_yielding(jinfo, yield);
        break;
      case MATMUL_MEMFILL_PCA:
        mmp_yielding(jinfo, yield);
        break;
      default:
        std::cout << "Unknown job type" << std::endl;
        break;
    }
    jinfo->s = COMPLETED;
    yield(&yield);
  }
}

void coro_rr(int id, job_info_t *&jinfo, coro_t::push_type &yield)
{
  LOG_PRINT(LOG_DEBUG, "[coro]: coro %d is ready!\n", id);
  yield(&yield);
  for(;;){
    switch(jinfo->jtype){
      case DESER_DECOMP_HASH:
        ddh_yielding_rr(jinfo, yield);
        break;
      case DECRYPT_MEMCPY_DP:
        dmdp_yielding_rr(jinfo, yield);
        break;
      case DECOMP_GATHER:
        decomp_gather_yielding_rr(jinfo, yield);
        break;
      case MEMCPY_GATHER:
        memcpy_gather_yielding_rr(jinfo, yield);
        break;
      case UPDATE_FILTER_HISTOGRAM:
        ufh_yielding_rr(jinfo, yield);
        break;
      case MATMUL_MEMFILL_PCA:
        mmp_yielding_rr(jinfo, yield);
        break;
      default:
        std::cout << "Unknown job type" << std::endl;
        break;
    }
    jinfo->s = COMPLETED;
    yield(&yield);
  }
}

void coro_rr_sw_fallback(int id, job_info_t *&jinfo, coro_t::push_type &yield)
{
  LOG_PRINT(LOG_DEBUG, "[coro]: coro %d is ready!\n", id);
  yield(&yield);
  for(;;){
    switch(jinfo->jtype){
      case DESER_DECOMP_HASH:
        ddh_yielding_rr_sw_fallback(jinfo, yield);
        break;
      case DECRYPT_MEMCPY_DP:
        dmdp_yielding_rr_sw_fallback(jinfo, yield);
        break;
      case DECOMP_GATHER:
        decomp_gather_yielding_rr_sw_fallback(jinfo, yield);
        break;
      case MEMCPY_GATHER:
        memcpy_gather_yielding_rr_sw_fallback(jinfo, yield);
        break;
      case UPDATE_FILTER_HISTOGRAM:
        ufh_yielding_rr_sw_fallback(jinfo, yield);
        break;
      case MATMUL_MEMFILL_PCA:
        mmp_yielding_rr_sw_fallback(jinfo, yield);
        break;
      default:
        std::cout << "Unknown job type" << std::endl;
        break;
    }
    jinfo->s = COMPLETED;
    yield(&yield);
  }
}

uint64_t get_free_hugepages() {
    std::ifstream file("/sys/devices/system/node/node1/hugepages/hugepages-1048576kB/free_hugepages");
    if (!file.is_open()) {
        std::cerr << "Error opening file to read free hugepages" << std::endl;
        return 0;
    }
    uint64_t free_hugepages;
    file >> free_hugepages;
    file.close();
    return free_hugepages;
}

void *workload_gen(void *arg){
  wrkld_gen_args_t *args = static_cast<wrkld_gen_args_t*>(arg);
  wrkload_ring_t *q = args->q;
  int num_reqs = args->num_reqs;
  int warm_up_reqs = num_reqs / 10;
  int server_node = args->server_node;
  uint64_t pushed = 0;
  uint64_t size = 16 * 1024;
  uint64_t num_samples = std::min(num_reqs, 1000);
  uint64_t sampling_interval = num_reqs / num_samples;
  pthread_barrier_t *start_barrier = args->start_barrier;
  uint64_t num_ddh_requests = args->num_ddh_reqs;
  uint64_t num_dmdp_requests = args->num_dmdp_reqs;
  uint64_t num_decomp_gather_requests = args->num_dg_reqs;
  uint64_t num_memcpy_gather_requests = args->num_mg_reqs;
  uint64_t num_ufh_requests = args->num_ufh_reqs;
  uint64_t num_mmp_requests = args->num_mmp_reqs;
  double peak = args->peak;
  double peak2 = args->peak2;
  if(num_ddh_requests < 1 && num_dmdp_requests < 1 &&
    num_decomp_gather_requests < 1 && num_memcpy_gather_requests < 1
    && num_ufh_requests < 1 && num_mmp_requests < 1){
    PRINT("No requests to inject\n");
    exit(1);
  }

  /* TODO: create_workload_payloads(ddh_bytes, workload_type) */
  uint64_t bytes = get_free_hugepages() * 1024 * 1024 * 1024;
  uint64_t comp_space_bytes = sizeof(idxd_comp) * num_reqs;
  uint64_t desc_space_bytes = sizeof(idxd_desc) * num_reqs;
  uint64_t args_space_bytes = (sizeof(ddh_args_t) * num_ddh_requests) + (sizeof(dmdp_args_t) * num_dmdp_requests) + (sizeof(decomp_gather_args) * num_decomp_gather_requests);
  uint64_t hdr_space_bytes = sizeof(hdr) * num_reqs;
  uint64_t necessary_bytes = args_space_bytes + comp_space_bytes + desc_space_bytes + hdr_space_bytes;
  if(necessary_bytes > bytes){
    LOG_PRINT(LOG_ERR, "Not enough memory to allocate all the necessary buffers\n");
    return NULL;
  }
  uint64_t ddh_bytes = (bytes - necessary_bytes); // TODO: mix workloads
  uint64_t dmdp_bytes = (bytes - necessary_bytes);
  uint64_t dg_bytes = (bytes - necessary_bytes);
  uint64_t mg_bytes = (bytes - necessary_bytes);
  uint64_t ufh_bytes = (bytes - necessary_bytes);
  uint64_t mmp_bytes = (bytes - necessary_bytes);

  /* per-request items */
  struct numa_mem *nm = (struct numa_mem *)calloc(1, sizeof(struct numa_mem));
  idxd_desc *desc = (idxd_desc *)alloc_numa_offset(nm, num_reqs * sizeof(idxd_desc), 0);
  idxd_comp *comp = (idxd_comp *)alloc_numa_offset(nm, num_reqs * sizeof(idxd_comp), 0);
  hdr *hdrs = (hdr *)alloc_numa_offset(nm, num_reqs * sizeof(hdr), 0);

  std::random_device rd;
  std::mt19937 gen(rd());

  dist_t dist_type = args->dist_type;
  std::exponential_distribution<> exp_dist(1.0 / peak);
  std::uniform_real_distribution<double> uni_dist(0.0, 1.0);
  std::function<double()> dist;
  if (dist_type == EXPONENTIAL) {
    PRINT("[Workload Gen] Exponential distribution\n");
    dist = [&exp_dist, &gen]() { return exp_dist(gen); };
  } else if (dist_type == BIMODAL) {
    PRINT("[Workload Gen] Bimodal distribution\n");
    dist = [&uni_dist, &gen, &peak, &peak2]() {
      double rand_val = uni_dist(gen);
      if (rand_val < 0.5) {
        return peak; // 1KB
      } else {
        return peak2; // 16KB
      }
    };
  } else if (dist_type == EXTREME_BIMODAL) {
    PRINT("[Workload Gen] Extreme Bimodal distribution\n");
    dist = [&uni_dist, &gen, &peak, &peak2]() {
      double rand_val = uni_dist(gen);
      if (rand_val < 0.995) {
        return peak; // 1KB
      } else {
        return peak2; // 16KB
      }
    };
  } else if (dist_type == DETERMINISTIC){
    PRINT("[Workload Gen] Deterministic distribution\n");
    dist = [&peak]() { return peak; };
  } else {
    PRINT("[Workload Gen] Uniform distribution\n");
    dist = [&uni_dist, &gen]() { return uni_dist(gen); };
  }

  std::vector<uint64_t> ser_buf_offsets;
  std::vector<uint64_t> payload_sizes;
  std::vector<uint64_t> max_comp_sizes;
  std::vector<uint64_t> max_payload_sizes;
  uint64_t used_payload_bytes = 0, total_ser_buf_space = 0, total_dbuf_bytes = 0;
  int s_sz;
  int num_ddh_bufs = 0;
  double target_ratio = 3.0;
  uint64_t decomp_out_space = IAA_DECOMPRESS_MAX_DEST_SIZE;
  for(;;){
    uint64_t rand_payload_size = static_cast<uint64_t>(dist());
    uint64_t decomp_size = rand_payload_size * target_ratio;
    // uint64_t decomp_size = std::max(static_cast<uint64_t>(768), std::min(rand_payload_size, static_cast<uint64_t>(2 * 1024 * 1024)));
    uint64_t max_comp_size = get_compress_bound(decomp_size);
    uint64_t max_expand_bytes = max_comp_size + MAX_SER_OVERHEAD_BYTES;
    max_expand_bytes = (max_expand_bytes + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
    used_payload_bytes += (max_expand_bytes + decomp_out_space);
    if (used_payload_bytes > ddh_bytes || num_ddh_bufs == num_ddh_requests){
      used_payload_bytes -= (max_expand_bytes + decomp_out_space);
      break;
    } else {
      max_comp_sizes.push_back(max_comp_size);
      max_payload_sizes.push_back(max_expand_bytes);
      ser_buf_offsets.push_back(total_ser_buf_space);
      payload_sizes.push_back(rand_payload_size);
      total_ser_buf_space += max_expand_bytes;
      total_dbuf_bytes += decomp_out_space;
      LOG_PRINT(LOG_DEBUG, "EXP: %lu SER: %lu TotalSer: %lu TotalDbuf: %lu\n",
        rand_payload_size, max_expand_bytes, total_ser_buf_space, total_dbuf_bytes);
      num_ddh_bufs++;
    }
  }

  int num_dmdp_bufs = 0;
  std::vector<uint64_t> enc_buf_offsets;
  std::vector<uint64_t> enc_payload_sizes;
  uint64_t total_enc_buf_space = 0;
  for(;;){
    uint64_t rand_payload_size = static_cast<uint64_t>(dist());
    rand_payload_size = std::max(rand_payload_size, static_cast<uint64_t>(256));
    rand_payload_size = std::min(rand_payload_size, static_cast<uint64_t>(2 * 1024 * 1024));
    uint64_t enc_size = rand_payload_size;
    used_payload_bytes += 3 * enc_size;
    if (used_payload_bytes > dmdp_bytes || num_dmdp_bufs == num_dmdp_requests){
      used_payload_bytes -= 3 * enc_size;
      break;
    } else {
      enc_buf_offsets.push_back(total_enc_buf_space);
      enc_payload_sizes.push_back(rand_payload_size);
      total_enc_buf_space += enc_size;
      LOG_PRINT(LOG_DEBUG, "EXP: %lu ENC: %lu TotalEnc: %lu\n",
        rand_payload_size, enc_size, total_enc_buf_space);
      num_dmdp_bufs++;
    }
  }

  int num_dg_bufs = 0;
  std::vector<uint64_t> comp_buf_offsets;
  std::vector<uint64_t> uncompbuf_offsets;
  std::vector<uint64_t> uncompbuf_sizes;
  std::vector<uint64_t> max_comp_sizes_2;
  uint64_t total_comp_buf_space = 0;
  uint64_t total_dbuf_space = 0;
  for(;;){
    uint64_t rand_payload_size = static_cast<uint64_t>(dist());
    uint64_t decomp_size = rand_payload_size * target_ratio;
    decomp_size = (decomp_size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
    if (decomp_size < 256) {
      decomp_size = 256;
    }
    if (decomp_size > 2 * 1024 * 1024) {
      decomp_size = 2 * 1024 * 1024;
    }
    uint64_t max_comp_size = get_compress_bound(decomp_size);
    max_comp_size = (max_comp_size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
    used_payload_bytes += (max_comp_size + decomp_out_space);
    if (used_payload_bytes > dg_bytes || num_dg_bufs == num_decomp_gather_requests){
      used_payload_bytes -= (max_comp_size + decomp_out_space);
      break;
    } else {
      max_comp_sizes_2.push_back(max_comp_size);
      uncompbuf_sizes.push_back(decomp_size);
      comp_buf_offsets.push_back(total_comp_buf_space);
      total_comp_buf_space += max_comp_size;
      total_dbuf_space += decomp_out_space;
      LOG_PRINT(LOG_DEBUG, "EXP: %lu DG: %lu TotalDG: %lu TotalDbuf: %lu\n",
        rand_payload_size, max_comp_size, total_comp_buf_space, total_dbuf_space);
      num_dg_bufs++;
    }
  }

  std::vector<uint64_t> memcpy_buf_offsets;
  std::vector<uint64_t> memcpy_payload_sizes;
  uint64_t total_memcpy_buf_space = 0;
  int num_memcpy_bufs = 0;
  for(;;){
    uint64_t rand_payload_size = static_cast<uint64_t>(dist());
    rand_payload_size = (rand_payload_size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
    rand_payload_size = std::max(rand_payload_size, static_cast<uint64_t>(256));
    rand_payload_size = std::min(rand_payload_size, static_cast<uint64_t>(4 * 1024 * 1024));
    if(used_payload_bytes + rand_payload_size > mg_bytes || num_memcpy_bufs == num_memcpy_gather_requests){
      break;
    } else {
      memcpy_buf_offsets.push_back(total_memcpy_buf_space);
      memcpy_payload_sizes.push_back(rand_payload_size);
      total_memcpy_buf_space += rand_payload_size;
      used_payload_bytes += (rand_payload_size * 2);
      LOG_PRINT(LOG_DEBUG, "EXP: %lu MG: %lu TotalMG: %lu\n",
        rand_payload_size, rand_payload_size, total_memcpy_buf_space);
      num_memcpy_bufs++;
    }
  }

  int num_ufh_bufs = 0;
  std::vector<uint64_t> upd_buf_offsets;
  std::vector<uint64_t> extracted_buf_offsets;
  std::vector<uint64_t> hist_buf_offsets;
  std::vector<uint64_t> scat_buf_offsets;
  std::vector<uint64_t> num_accesses;
  std::vector<uint64_t> aecs_offsets;
  uint64_t total_upd_buf_space = 0;
  uint64_t total_extracted_buf_space = 0;
  uint64_t total_hist_buf_space = 0;
  uint64_t total_scat_buf_space = 0;
  uint64_t total_aecs_space = 0;
  uint64_t total_arg_space = 0;
  for(;;){
    uint64_t rand_payload_size = static_cast<uint64_t>(dist());
    rand_payload_size = (rand_payload_size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
    rand_payload_size = std::max(rand_payload_size, static_cast<uint64_t>(256));
    rand_payload_size = std::min(rand_payload_size, static_cast<uint64_t>(4 * 1024 * 1024));
    if( used_payload_bytes +
          rand_payload_size +
          IAA_FILTER_MAX_DEST_SIZE +
          (256 * 3 * sizeof(int)) +
          ((rand_payload_size / CACHE_LINE_SIZE) * sizeof(int)) +
          CACHE_LINE_SIZE > ufh_bytes
          || num_ufh_bufs == num_ufh_requests){
      PRINT("Used: %lu\n", used_payload_bytes);
      break;
    } else {
      upd_buf_offsets.push_back(total_upd_buf_space);
      extracted_buf_offsets.push_back(total_extracted_buf_space);
      hist_buf_offsets.push_back(total_hist_buf_space);
      scat_buf_offsets.push_back(total_scat_buf_space);
      aecs_offsets.push_back(total_aecs_space);
      num_accesses.push_back(rand_payload_size / CACHE_LINE_SIZE);
      total_upd_buf_space += rand_payload_size;
      total_extracted_buf_space += IAA_FILTER_MAX_DEST_SIZE;
      total_hist_buf_space += ( 256 * 3 * sizeof(int));
      total_scat_buf_space += (rand_payload_size / CACHE_LINE_SIZE) * sizeof(int);
      total_aecs_space += CACHE_LINE_SIZE; /*aecs is smaller than 64B*/
      total_arg_space += sizeof(ufh_args_t);
      used_payload_bytes = total_upd_buf_space + total_extracted_buf_space +
        total_hist_buf_space + total_scat_buf_space + total_aecs_space + total_arg_space;
      LOG_PRINT(LOG_DEBUG, "EXP: %lu TotalUFH: %lu\n",
        rand_payload_size, total_upd_buf_space +
        total_extracted_buf_space +
        total_hist_buf_space +
        total_scat_buf_space +
        total_aecs_space);
      num_ufh_bufs++;
    }
  }

  int num_mmp_bufs = 0;
  uint64_t total_mm_data_bytes = 0;
  uint64_t total_mat_bytes = 0;
  uint64_t total_mean_vector_bytes = 0;
  uint64_t total_mmp_args_bytes = 0;
  uint64_t mat_bytes = 0;
  std::vector<uint64_t> mat_byte_sizes;
  std::vector<int> mat_lens;
  for(;;){
    uint64_t payload_size = static_cast<uint64_t>(dist());
    uint64_t matrix_len = floor(sqrt(payload_size/sizeof(int)));
    if(matrix_len < 16){
      matrix_len = 16;
    } else if (matrix_len > 1024){
      matrix_len = 1024;
    }
    uint64_t mat_size_bytes = matrix_len * matrix_len * sizeof(int);
    if (used_payload_bytes
          + sizeof(mm_data_t)
          + 3 * mat_size_bytes
          + matrix_len * sizeof(int) > mmp_bytes
          || num_mmp_bufs == num_mmp_requests){
      break;
    } else {
      total_mm_data_bytes += sizeof(mm_data_t);
      total_mat_bytes += 3 * mat_size_bytes;
      mat_bytes += mat_size_bytes;
      total_mean_vector_bytes += matrix_len * sizeof(int);
      total_mmp_args_bytes += sizeof(mmpc_args_t);
      mat_byte_sizes.push_back(mat_size_bytes);
      mat_lens.push_back(matrix_len);
      used_payload_bytes =
        total_mm_data_bytes + total_mat_bytes +
        total_mean_vector_bytes + total_mmp_args_bytes;
      LOG_PRINT(LOG_DEBUG, "EXP: %lu TotalMMP: %lu\n",
        payload_size,
        used_payload_bytes);
      num_mmp_bufs++;
    }
  }


  LOG_PRINT(LOG_DEBUG, "NumBufs: %d\n", num_dg_bufs + num_ddh_bufs + num_dmdp_bufs + num_memcpy_bufs + num_ufh_bufs);
  char *s_bufs = (char *)alloc_numa_offset(nm, total_ser_buf_space, 0);
  char *d_bufs = (char *)alloc_numa_offset(nm, total_dbuf_bytes, 0);
  ddh_args_t *ddh_args = (ddh_args_t *)alloc_numa_offset(nm, num_ddh_requests * sizeof(ddh_args_t), 0);

  char *enc_bufs = (char *)alloc_numa_offset(nm, total_enc_buf_space, 0);
  char *dec_bufs = (char *)alloc_numa_offset(nm, total_enc_buf_space, 0);
  char *dst_bufs = (char *)alloc_numa_offset(nm, total_enc_buf_space, 0);
  dmdp_args_t *dmdp_args = (dmdp_args_t *)alloc_numa_offset(nm, num_dmdp_requests * sizeof(dmdp_args_t), 0);

  char *dg_bufs = (char *)alloc_numa_offset(nm, total_comp_buf_space, 0);
  char *dbufs = (char *)alloc_numa_offset(nm, total_dbuf_space, 0);
  struct decomp_gather_args *dg_args = (struct decomp_gather_args *)alloc_numa_offset(nm, num_decomp_gather_requests * sizeof(struct decomp_gather_args), 0);

  char *mg_bufs = (char *)alloc_numa_offset(nm, total_memcpy_buf_space, 0);
  char *chase_bufs = (char *)alloc_numa_offset(nm, total_memcpy_buf_space, 0);
  struct decomp_gather_args *mg_args = (struct decomp_gather_args *)alloc_numa_offset(nm, num_memcpy_gather_requests * sizeof(struct decomp_gather_args), 0);

  char *upd_bufs = (char *)alloc_numa_offset(nm, total_upd_buf_space, 0);
  char *extracted_bufs = (char *)alloc_numa_offset(nm, total_extracted_buf_space, 0);
  char *hist_bufs = (char *)alloc_numa_offset(nm, total_hist_buf_space, 0);
  char *scat_bufs = (char *)alloc_numa_offset(nm, total_scat_buf_space, 0);
  char *aecs_bufs = (char *)alloc_numa_offset(nm, total_aecs_space, 0);
  ufh_args_t *ufh_args = (ufh_args_t *)alloc_numa_offset(nm, num_ufh_requests * sizeof(ufh_args_t), 0);

  mm_data_t *mm_data = (mm_data_t *)alloc_numa_offset(nm, total_mm_data_bytes, 0);
  int *mat_a = (int *)alloc_numa_offset(nm, mat_bytes, 0);
  int *mat_b = (int *)alloc_numa_offset(nm, mat_bytes, 0);
  int *mat_c = (int *)alloc_numa_offset(nm, mat_bytes, 0);
  int *mean_vector = (int *)alloc_numa_offset(nm, total_mean_vector_bytes, 0);
  mmpc_args_t *mmpc_args = (mmpc_args_t *)alloc_numa_offset(nm, total_mmp_args_bytes, 0);

  int rc = alloc_numa_mem(nm, PAGE_SIZE, server_node);
  add_base_addr(nm, (void **)&desc);
  add_base_addr(nm, (void **)&comp);
  add_base_addr(nm, (void **)&s_bufs);
  add_base_addr(nm, (void **)&d_bufs);
  add_base_addr(nm, (void **)&ddh_args);
  add_base_addr(nm, (void **)&hdrs);

  add_base_addr(nm, (void **)&enc_bufs);
  add_base_addr(nm, (void **)&dec_bufs);
  add_base_addr(nm, (void **)&dst_bufs);
  add_base_addr(nm, (void **)&dmdp_args);

  add_base_addr(nm, (void **)&dg_bufs);
  add_base_addr(nm, (void **)&dbufs);
  add_base_addr(nm, (void **)&dg_args);

  add_base_addr(nm, (void **)&mg_bufs);
  add_base_addr(nm, (void **)&chase_bufs);
  add_base_addr(nm, (void **)&mg_args);

  add_base_addr(nm, (void **)&upd_bufs);
  add_base_addr(nm, (void **)&extracted_bufs);
  add_base_addr(nm, (void **)&hist_bufs);
  add_base_addr(nm, (void **)&scat_bufs);
  add_base_addr(nm, (void **)&aecs_bufs);
  add_base_addr(nm, (void **)&ufh_args);

  add_base_addr(nm, (void **)&mm_data);
  add_base_addr(nm, (void **)&mat_a);
  add_base_addr(nm, (void **)&mat_b);
  add_base_addr(nm, (void **)&mat_c);
  add_base_addr(nm, (void **)&mean_vector);
  add_base_addr(nm, (void **)&mmpc_args);

  memset((void *)comp, 0, num_reqs * sizeof(idxd_comp));
  memset((void *)desc, 0, num_reqs * sizeof(idxd_desc));

  std::vector<uint64_t> ser_sizes;
  for(int i=0; i<num_ddh_requests; i++){
    uint64_t ser_buf_offset = ser_buf_offsets[(i % num_ddh_bufs)];
    uint64_t max_payload_expansion = max_payload_sizes[(i % num_ddh_bufs)];
    uint64_t payload_size = payload_sizes[(i % num_ddh_bufs)];
    uint64_t max_comp_size = max_comp_sizes[(i % num_ddh_bufs)];
    gen_ser_comp_payload(
      (void *)(s_bufs + ser_buf_offset),
      payload_size,
      max_comp_size,
      max_payload_expansion,
      &s_sz,
      target_ratio);
    ser_sizes.push_back(s_sz);
    ddh_args[i].desc = &desc[i];
    ddh_args[i].comp = &comp[i];
    ddh_args[i].s_buf = (char *)(s_bufs + ser_buf_offset);
    ddh_args[i].s_sz = s_sz;
    ddh_args[i].d_buf = &d_bufs[(i % num_ddh_bufs) * decomp_out_space];
    ddh_args[i].d_sz = payload_size * target_ratio;
    ddh_args[i].id = i;
    ddh_args[i].p_off = 0;
  }

  double mean = std::accumulate(payload_sizes.begin(), payload_sizes.end(), 0.0) / payload_sizes.size();
  PRINT("[Workload Gen] Exponential Workload Distribution DDH Attempted Payload Size: %f\n", mean);
  double actual_mean = std::accumulate(ser_sizes.begin(), ser_sizes.end(), 0.0) / ser_sizes.size();
  PRINT("[Workload Gen] Actual Payload Size: %f\n", actual_mean);

  for(int i=0; i<num_dmdp_requests; i++){
    uint64_t enc_buf_offset = enc_buf_offsets[(i % num_dmdp_bufs)];
    uint64_t enc_size = enc_payload_sizes[(i % num_dmdp_bufs)];
    char *tmp_plain_buf = (char *)malloc(enc_size);
    memset_pattern((void *)tmp_plain_buf, 0xdeadbeef, enc_size);
    enc_buf((Ipp8u *)(enc_bufs + enc_buf_offset), (Ipp8u *)tmp_plain_buf, enc_size);
    dmdp_args[i].desc = &desc[num_ddh_requests + i];
    dmdp_args[i].comp = &comp[num_ddh_requests + i];
    dmdp_args[i].enc_buf = (char *)(enc_bufs + enc_buf_offset);
    dmdp_args[i].dec_buf = (char *)(dec_bufs + enc_buf_offset);
    dmdp_args[i].dst_buf = (char *)(dst_bufs + enc_buf_offset);
    dmdp_args[i].score = 0.0;
    dmdp_args[i].sz = enc_size;
    dmdp_args[i].id = i;
    dmdp_args[i].p_off = 0;
    free(tmp_plain_buf);
  }

  double mean_enc = std::accumulate(enc_payload_sizes.begin(), enc_payload_sizes.end(), 0.0) / enc_payload_sizes.size();
  PRINT("[Workload Gen] Exponential Workload Distribution DMDP Attempted Payload Size: %f\n", mean_enc);
  double actual_mean_enc = std::accumulate(enc_payload_sizes.begin(), enc_payload_sizes.end(), 0.0) / enc_payload_sizes.size();
  PRINT("[Workload Gen] Actual Payload Size: %f\n", actual_mean_enc);

  thread_wq_alloc(&m_iaa, 5);
  if(num_dg_bufs < num_decomp_gather_requests){
    PRINT("Not enough buffers for decomp gather... Dropping request to %d\n", num_dg_bufs);
    num_reqs = num_dg_bufs ;
    num_decomp_gather_requests = num_dg_bufs ;
    // exit(1);
  }
  for(int i=0; i<num_decomp_gather_requests; i++){
    uint64_t dg_buf_offset = comp_buf_offsets[(i % num_dg_bufs)];
    uint64_t max_comp_size = max_comp_sizes_2[(i % num_dg_bufs)];
    uLong uncomp_size = uncompbuf_sizes[(i % num_dg_bufs)];
    uint8_t *temp_comp_buf = (uint8_t *)malloc(max_comp_size);
    uint8_t *temp_uncomp_buf2 = (uint8_t *)malloc(uncomp_size);
    int comp_size = max_comp_size;
    lzdg_generate_reuse_buffers(
      temp_uncomp_buf2,
      uncomp_size,
      3.0, 3.0, 3.0
    );
    create_random_chain_starting_at(
      uncomp_size,
      (void **)temp_uncomp_buf2,
      (void **)(dbufs + (i % num_dg_bufs) * decomp_out_space)
    );
    int rc = gpcore_do_compress(
      (void *)(dg_bufs + dg_buf_offset),
      (void *)temp_uncomp_buf2,
      uncomp_size,
      &comp_size
    );
    if (rc != 0) {
      LOG_PRINT(LOG_ERR, "Compression for request:%d failed: %d\n", i, rc);
      exit(1);
    }
    // rc = gpcore_do_decompress(
    //   (void *)(dbufs + (i % num_dg_bufs) * decomp_out_space),
    //   (void *)(dg_bufs + dg_buf_offset),
    //   comp_size,
    //   &uncomp_size
    // );
    // if(rc != 0){
    //   LOG_PRINT(LOG_ERR, "Decompression for request:%d failed: %d\n", i, rc);
    //   exit(1);
    // }
    // debug_chain((void **)(dbufs + (i % num_dg_bufs) * decomp_out_space));
    dg_args[i].desc = &desc[num_ddh_requests + num_dmdp_requests + i];
    dg_args[i].comp = &comp[num_ddh_requests + num_dmdp_requests + i];
    dg_args[i].s_buf = (char *)(dg_bufs + dg_buf_offset);
    dg_args[i].d_buf = (float *)(&dbufs[(i % num_dg_bufs) * decomp_out_space]);
    dg_args[i].d_sz = uncomp_size;
    dg_args[i].s_sz = comp_size;
    dg_args[i].id = i;
    dg_args[i].p_off = 0;

    prepare_iaa_decompress_desc_with_preallocated_comp(
      &desc[num_ddh_requests + num_dmdp_requests + i],
      (uint64_t)(dg_bufs + dg_buf_offset),
      (uint64_t)(&dbufs[(i % num_dg_bufs) * decomp_out_space]),
      (uint64_t)(&comp[num_ddh_requests + num_dmdp_requests + i]),
      comp_size
    );
    while(enqcmd((void *)((char *)(m_iaa->wq_reg) + 0), &desc[num_ddh_requests + num_dmdp_requests + i]) ){
    }
    while(comp[num_ddh_requests + num_dmdp_requests + i].status == IAX_COMP_NONE){
    }
    if(comp[num_ddh_requests + num_dmdp_requests + i].status != IAX_COMP_SUCCESS && comp[num_ddh_requests + num_dmdp_requests + i].status != IAX_COMP_NONE){
      LOG_PRINT(LOG_ERR, "Decompression for request:%d failed: %d\n", i, comp[num_ddh_requests + num_dmdp_requests + i].status);
      exit(1);
    }
    chase_pointers(
      (void **)(&dbufs[(i % num_dg_bufs) * decomp_out_space]),
      uncomp_size/64
    );
    free(temp_uncomp_buf2);
    // uLong decomp_size = IAA_DECOMPRESS_MAX_DEST_SIZE;
    // rc = gpcore_do_decompress(
    //   (void *)(&dbufs[(i % num_dg_bufs) * decomp_out_space]),
    //   (void *)(dg_bufs + dg_buf_offset),
    //   comp_size,
    //   &decomp_size
    // );
    // if(rc != 0){
    //   LOG_PRINT(LOG_ERR, "Decompression for request:%d failed: %d\n", i, rc);
    //   exit(1);
    // }
    // chase_pointers((void **)(&dbufs[(i % num_dg_bufs) * decomp_out_space]), decomp_size/64);

  }
  // exit(1);
  double mean_dg = std::accumulate(uncompbuf_sizes.begin(), uncompbuf_sizes.end(), 0.0) / uncompbuf_sizes.size();
  PRINT("[Workload Gen] Exponential Workload Distribution DG Attempted Payload Size: %f\n", mean_dg);
  double actual_mean_dg = std::accumulate(uncompbuf_sizes.begin(), uncompbuf_sizes.end(), 0.0) / uncompbuf_sizes.size();
  PRINT("[Workload Gen] Actual Payload Size: %f\n", actual_mean_dg);

  thread_wq_alloc(&m_dsa, 4);
  if(num_memcpy_bufs < num_memcpy_gather_requests){
    LOG_PRINT(LOG_ERR, "Not enough buffers for memcpy gather\n");
    exit(1);
  }
  for(int i=0; i<num_memcpy_gather_requests; i++){
    uint64_t mg_buf_offset = memcpy_buf_offsets[(i % num_memcpy_bufs)];
    uint64_t mg_size = memcpy_payload_sizes[(i % num_memcpy_bufs)];
    char *src_ptr_buf = &(mg_bufs[mg_buf_offset]);
    create_random_chain_starting_at(mg_size, (void **)src_ptr_buf, (void **)(chase_bufs + mg_buf_offset));
    mg_args[i].s_buf = src_ptr_buf;
    mg_args[i].d_buf = (float *)(chase_bufs + mg_buf_offset);
    mg_args[i].s_sz = mg_size;
    mg_args[i].id = i;

    mg_args[i].p_off = 0;
    mg_args[i].desc = &desc[num_ddh_requests + num_dmdp_requests + num_decomp_gather_requests + i];
    mg_args[i].comp = &comp[num_ddh_requests + num_dmdp_requests + num_decomp_gather_requests + i];
    mg_args[i].d_sz = mg_size;
    mg_args[i].s_sz = mg_size;
    mg_args[i].id = i;
    mg_args[i].p_off = 0;

  }
  double mean_mg = std::accumulate(memcpy_payload_sizes.begin(), memcpy_payload_sizes.end(), 0.0) / memcpy_payload_sizes.size();
  PRINT("[Workload Gen] Exponential Workload Distribution MG Attempted Payload Size: %f\n", mean_mg);
  double actual_mean_mg = std::accumulate(memcpy_payload_sizes.begin(), memcpy_payload_sizes.end(), 0.0) / memcpy_payload_sizes.size();
  PRINT("[Workload Gen] Actual Payload Size: %f\n", actual_mean_mg);

  thread_wq_alloc(&m_iaa, 5);
  if(num_ufh_bufs < num_ufh_requests){
    PRINT("Not enough buffers for ufh... Dropping request to %d\n", num_ufh_bufs);
    num_reqs = num_ufh_bufs ;
    num_ufh_requests = num_ufh_bufs ;
    // exit(1);
  }

  for(int i=0; i<num_ufh_requests; i++){
    uint64_t upd_buf_offset = upd_buf_offsets[(i % num_ufh_bufs)];
    uint64_t extracted_buf_offset = extracted_buf_offsets[(i % num_ufh_bufs)];
    uint64_t hist_buf_offset = hist_buf_offsets[(i % num_ufh_bufs)];
    uint64_t scat_buf_offset = scat_buf_offsets[(i % num_ufh_bufs)];
    uint64_t aecs_offset = aecs_offsets[(i % num_ufh_bufs)];
    uint64_t num_access = num_accesses[(i % num_ufh_bufs)];

    uint64_t payload_size = num_access * CACHE_LINE_SIZE;
    float *this_upd_buf = (float *)(upd_bufs + upd_buf_offset);
    // memset_pattern((void *)this_upd_buf, 0xabcdbeef, payload_size);
    memset((void *)this_upd_buf, 10, payload_size);
    ufh_args[i].desc = &desc[num_ddh_requests + num_dmdp_requests + num_decomp_gather_requests + num_memcpy_gather_requests + i];
    ufh_args[i].comp = &comp[num_ddh_requests + num_dmdp_requests + num_decomp_gather_requests + num_memcpy_gather_requests + i];
    ufh_args[i].upd_buf = this_upd_buf;
    ufh_args[i].extracted = (uint8_t *)(extracted_bufs + extracted_buf_offset);
    ufh_args[i].hist = (uint8_t *)(hist_bufs + hist_buf_offset);
    ufh_args[i].scat_buf = (int *)(scat_bufs + scat_buf_offset);
    ufh_args[i].aecs = (uint8_t *)(aecs_bufs + aecs_offset);
    ufh_args[i].num_acc = num_access;
    ufh_args[i].low_val = 0;
    ufh_args[i].high_val = 256;

    ufh_args[i].id = i;
    ufh_args[i].p_off = 0;


  }

  if(num_mmp_bufs < num_mmp_requests){
    PRINT("Not enough buffers for mmp... Dropping request to %d\n", num_mmp_bufs);
    num_reqs = num_mmp_bufs ;
    num_mmp_requests = num_mmp_bufs ;
    // exit(1);
  }

  for(int i=0; i<num_mmp_requests; i++){
    uint64_t mm_data_offset = 0;
    uint64_t mat_a_offset = 0;
    uint64_t mat_b_offset = 0;
    uint64_t mat_c_offset = 0;
    uint64_t mean_vector_offset = 0;

    int mmpc_req_offset = num_ddh_requests +
                          num_dmdp_requests +
                          num_decomp_gather_requests +
                          num_memcpy_gather_requests +
                          num_ufh_requests +
                          i;

    int *m_mat_a = (int *)(mat_a + mat_a_offset);
    int *m_mat_b = (int *)(mat_b + mat_b_offset);
    int *m_mat_c = (int *)(mat_c + mat_c_offset);
    mm_data_t *m_mm_data = (mm_data_t *)(mm_data + mm_data_offset);
    int *m_mean_vector = (int *)(mean_vector + mean_vector_offset);
    int m_mat_size_bytes = mat_byte_sizes[(i % num_mmp_bufs)];
    int m_mat_len = mat_lens[(i % num_mmp_bufs)];

    memset((void *)m_mat_c, 0, m_mat_size_bytes);

    for(int k=0; k<m_mat_len; k++){
      for(int j=0; j<m_mat_len; j++){
        if(k == j){
          m_mat_a[k*m_mat_len + j] = 1;
          m_mat_b[k*m_mat_len + j] = 1;
        } else {
          m_mat_a[k*m_mat_len + j] = 0;
          m_mat_b[k*m_mat_len + j] = 0;
        }
      }
    }

    mmpc_args[i].desc = &desc[mmpc_req_offset];
    mmpc_args[i].comp = &comp[mmpc_req_offset];
    mmpc_args[i].mm_data = m_mm_data;
    mmpc_args[i].mm_data->matrix_A = m_mat_a;
    mmpc_args[i].mm_data->matrix_B = m_mat_b;
    mmpc_args[i].mm_data->matrix_out = m_mat_c;
    mmpc_args[i].mat_size_bytes = m_mat_size_bytes;
    mmpc_args[i].mm_data->matrix_len = m_mat_len;
    mmpc_args[i].mean_vec = m_mean_vector;
    mmpc_args[i].id = i;
    mmpc_args[i].p_off = 0;

  }

  void **arg_list = (void **)malloc(num_reqs * sizeof(void *));
  job_type *job_list = (job_type *)malloc(num_reqs * sizeof(job_type));
  if (num_ddh_requests > 0) {
    for (int i = 0; i < num_reqs; i++) {
      arg_list[i] = (void *)&ddh_args[i % num_ddh_requests];
      job_list[i] = DESER_DECOMP_HASH;
    }
    num_requests = num_ddh_requests;
  } else if (num_dmdp_requests > 0) {
    for (int i = 0; i < num_reqs; i++) {
      arg_list[i] = (void *)&dmdp_args[i % num_dmdp_requests];
      job_list[i] = DECRYPT_MEMCPY_DP;
    }
    num_requests = num_dmdp_requests;
  } else if (num_decomp_gather_requests > 0) {
    for (int i = 0; i < num_reqs; i++) {
      arg_list[i] = (void *)&dg_args[i % num_decomp_gather_requests];
      job_list[i] = DECOMP_GATHER;
    }
    num_requests = num_decomp_gather_requests;
  } else if (num_memcpy_gather_requests > 0){
    for(int i=0; i<num_reqs; i++){
      arg_list[i] = (void *)&mg_args[i % num_memcpy_gather_requests];
      job_list[i] = MEMCPY_GATHER;
    }
    num_requests = num_memcpy_gather_requests;
  } else if (num_ufh_requests > 0){
    for(int i=0; i<num_reqs; i++){
      arg_list[i] = (void *)&ufh_args[i % num_ufh_requests];
      job_list[i] = UPDATE_FILTER_HISTOGRAM;
    }
    num_requests = num_ufh_requests;
  } else if (num_mmp_requests > 0){
    for(int i=0; i<num_reqs; i++){
      arg_list[i] = (void *)&mmpc_args[i % num_mmp_requests];
      job_list[i] = MATMUL_MEMFILL_PCA;
    }
    num_requests = num_mmp_requests;
  } else {
    std::cerr << "No requests to inject" << std::endl;
    exit(1);
  }

  #ifdef POISSON
  uint64_t *offsets = (uint64_t *)malloc(num_reqs * sizeof(uint64_t));
  uint64_t freqHz = 2100000000;
  double max_rps = 1000000;
  double load = args->load;
  PRINT("[Workload Gen] Load: %f MRPS: %f\n", load, load * max_rps);
  gen_interarrival_poisson(max_rps * load, offsets, num_reqs, freqHz);
  for(int i=0; i<num_reqs; i++){
    void *arg = arg_list[i];
    hdrs[i].arrival = offsets[i];
    hdrs[i].id = i;
    hdrs[i].req_type = job_list[i];
    hdrs[i].w_args = arg;
    #if defined(LOST_ENQ_TIME) || defined(COUNT_LOST_ENQ)
    hdrs[i].failed_enqs = 0;
    #endif
  }
  workload_start_cycle = 0;
  #endif

  // if(args->unloaded){
    job_info_t job_info;
    coro_info_t coro_info;
    coro_info.coro = new coro_t::pull_type(
      boost::bind(coro_blocking, 0, &job_info, _1)
    );
    uint64_t *starts, *ends, start, end;
    thread_wq_alloc(&m_dsa, 4);
    thread_wq_alloc(&m_iaa, 5);
    starts = (uint64_t *)malloc(num_reqs * sizeof(uint64_t));
    ends = (uint64_t *)malloc(num_reqs * sizeof(uint64_t));
    for(int i=0; i<num_reqs; i++){
      start = fenced_rdtscp();
      job_info.args = arg_list[i];
      job_info.jtype = job_list[i];
      job_info.s = INIT;
      while(job_info.s != COMPLETED){
        (*(coro_info.coro))();
      }
      end = fenced_rdtscp();
      hdrs[i].unloaded = end - start;
      starts[i] = start;
      ends[i] = end;
    }
    double total_time = 0.0;
    for (int i = 0; i < num_reqs; i++) {
      total_time += (ends[i] - starts[i]);
    }
    double average_time = total_time / num_reqs;
    PRINT("[Workload Gen] Average Time per Request: %f\n", average_time);
    free(starts);
    free(ends);
  // }

  PRINT("[Workload Gen] Ready to Dispatch: %d\n", warm_up_reqs);

  pthread_barrier_wait(start_barrier);
  #ifdef BOTTLENECK
  uint64_t failed_enq = 0;
  #endif
  #ifdef LATENCY
  uint64_t tagged_jobs = 0;
  #endif
  #ifdef POISSON
  uint64_t now;
  uint64_t fst_ts = rdtsc();
  #endif
  /* warm up */

  while(pushed < warm_up_reqs){
    #ifdef POISSON
    do{
      now = rdtsc();
    } while(now - fst_ts < offsets[pushed]);
    #endif
    while(!q->try_enqueue(hdrs[pushed]));
    pushed++;
  }

  #if defined(BOTTLENECK) || defined(THROUGHPUT)
  start = fenced_rdtscp();
  #endif
  while(pushed < num_reqs){
    #ifdef LATENCY
    if( pushed % sampling_interval == 0){
      hdrs[pushed].tagged = true;
      hdrs[pushed].injected = rdtsc();
      tagged_jobs++;
    } else {
      hdrs[pushed].tagged = false;
    }
    #else
    // hdr h {(uint32_t)pushed, DESER_DECOMP_HASH, arg};
    hdr h {pushed, DECRYPT_MEMCPY_DP, arg};
    #endif
    #ifdef POISSON
    do{
      now = rdtsc();
    } while(now - fst_ts < offsets[pushed]);
    #endif
    #ifdef LATENCY
    while(!q->try_enqueue(hdrs[pushed]));
    #else
    while(!q->try_enqueue(h));
    #endif
    pushed++;
  }
  #if defined(BOTTLENECK) || defined(THROUGHPUT)
  end = fenced_rdtscp();
  PRINT("[Workload Gen] Enqueue Rate (MRPS): %f\n", (double)(num_reqs-warm_up_reqs) / ((double)(end - start) / 2100));
  #endif


  #ifdef DEBUG
  PRINT("Workload gen pushed %ld requests\n", pushed);
  #endif


  #ifdef LATENCY
  PRINT("[Workload Gen] Sampling Interval: %lu\n", sampling_interval);
  PRINT("[Workload Gen] Mean/Peak: %f\n", peak);
  PRINT("[Workload Gen] Tagged Jobs: %ld\n", tagged_jobs);
  workload_start_cycle = fst_ts;
  #endif

  pthread_barrier_wait(args->exit_barrier);
  return NULL;
}

#ifdef JSQ
bool worker_info_ptr_cmp(const worker_info_t* ptr1, const worker_info_t* ptr2) {
	return *ptr1 < *ptr2;
}
#endif

void *dispatcher(void *arg){
  dispatch_args_t *args = static_cast<dispatch_args_t*>(arg);
  dispatch_ring_t **d = args->wqs;
  wrkload_ring_t *q = args->inq;
  int num_workers = args->num_workers;
  int popped = 0, pushed = 0;
  int num_reqs = args->num_requests;
  int outq;
  pthread_barrier_t *start_barrier = args->start_barrier;
  int running_jobs;
  hdr h;

  #ifdef JSQ
  worker_info_t *tmp_w;
  int **pp_finished_jobs = args->pp_finished_jobs;
  std::priority_queue<worker_info_t*, std::vector<worker_info_t*>,
    decltype(&worker_info_ptr_cmp)>
      worker_queue(worker_info_ptr_cmp);
  for(int i=0; i<num_workers; i++){
    worker_info_t *w =
      (worker_info_t *)calloc(1, sizeof(worker_info_t));
    w->dq_idx = i;
    w->num_running_jobs = 0;
    w->dispatched_jobs = 0;
    w->p_finished_jobs = pp_finished_jobs[i];
    worker_queue.push(w);
  }
  #endif

  while(num_requests.load() == 0){
    _mm_pause();
  }
  num_reqs = num_requests.load();

  pthread_barrier_wait(start_barrier);
  #ifdef LATENCY
  uint64_t tagged_jobs=0;
  #endif
  #if defined(BOTTLENECK) || defined(THROUGHPUT) || defined(LATENCY)
  uint64_t warmup = num_reqs / 10;
  uint64_t start, end;
  while(popped < warmup){
    if (q->try_dequeue(h)){
      #if defined(LATENCY)
      if(h.tagged){
        h.dispatched = fenced_rdtscp();
        tagged_jobs++;
      }
      #endif

      #ifdef JSQ
      tmp_w = worker_queue.top();
      worker_queue.pop();
      outq = tmp_w->dq_idx;
      tmp_w->num_running_jobs =
        (tmp_w->dispatched_jobs++) - *(tmp_w->p_finished_jobs) + 1;
      worker_queue.push(tmp_w);
      #else
      outq = popped % num_workers;
      #endif
      while (!d[outq]->try_enqueue(h)){
        outq = (outq + 1) % num_workers;
      }
      popped++;
    }
  }
  start = fenced_rdtscp();
  #endif
  /* Dequeue from the workload ring and print the requests */
  while(popped < num_requests){
    if (q->try_dequeue(h)){

      #if defined(LATENCY)
      if(h.tagged){
        h.dispatched = fenced_rdtscp();
        tagged_jobs++;
      }
      #endif


        #ifdef JSQ
      tmp_w = worker_queue.top();
      worker_queue.pop();
      outq = tmp_w->dq_idx;
      tmp_w->num_running_jobs =
        (tmp_w->dispatched_jobs++) - *(tmp_w->p_finished_jobs) + 1;
      worker_queue.push(tmp_w);
        #else
      outq = popped % num_workers;
        #endif
      while (!d[outq]->try_enqueue(h)) {
        outq = (outq + 1) % num_workers;
      }
      popped++;
    }
  }
  #if defined(BOTTLENECK) || defined(THROUGHPUT)
  end = fenced_rdtscp();
  PRINT("[Dispatcher] Forward Rate (MRPS): %f\n", (double)(num_requests-warmup) / ((double)(end - start) / 2100));
  #endif
  #if defined(LATENCY)
  PRINT("[Dispatcher] Tagged Jobs: %ld\n", tagged_jobs);
  #endif

  pthread_barrier_wait(args->exit_barrier);
  return NULL;
}

void *worker_rr(void *arg){
  wrk_args_t *args = static_cast<wrk_args_t*>(arg);
  dispatch_ring_t *q = args->q;
  response_ring_t *c_q = args->c_queue;
  int id = args->id;
  int popped = 0;
  int num_requests = args->num_requests;
  int num_coros = args->num_coros;
  struct numa_mem *nm;
  nm = (struct numa_mem *)calloc(1, sizeof(struct numa_mem));
  int node = args->node;
  int iaa_dev_id = args->iaa_dev_id;
  int dsa_dev_id = args->dsa_dev_id;
  pthread_barrier_t *start_barrier = args->start_barrier;
  hdr h;

  std::vector<coro_info_t*> idle_coros;
  idle_coros.reserve(num_coros);

  coro_t::pull_type *coros
    = (coro_t::pull_type *)malloc(num_coros * sizeof(coro_t::pull_type));
  coro_info_t *c_infos
    = (coro_info_t *)malloc(num_coros * sizeof(coro_info_t));
  job_info_t *job_infos = args->jinfos;

  std::deque<coro_info_t*> busy_coros;


  char *st_pool = args->stack_pool;
  /* initialize coros */
  for(int i = 0; i < num_coros; i++){
    coros[i] =
      coro_t::pull_type(
          SimpleStack(static_cast<char*>(st_pool + (i * STACK_SIZE))),
          boost::bind( coro_rr, i, &job_infos[i], _1));

    c_infos[i].coro = &coros[i];
    c_infos[i].yield = static_cast<coro_t::push_type*>(coros[i].get());
    c_infos[i].jinfo = &job_infos[i];
    c_infos[i].jinfo->s = INIT;
    c_infos[i].jinfo->failed_enq = 0;
    idle_coros.push_back(&c_infos[i]);
  }

  LOG_PRINT(LOG_DEBUG, "Worker %d allocating IAA %d\n", id, iaa_dev_id);
  thread_wq_alloc(&m_iaa, iaa_dev_id);
  LOG_PRINT(LOG_DEBUG, "Worker %d allocating DSA %d\n", id, dsa_dev_id);
  thread_wq_alloc(&m_dsa, dsa_dev_id);

  #ifdef JSQ
  int *p_finished_jobs = args->p_finished_jobs;
  *p_finished_jobs = 0;
  #endif

  pthread_barrier_wait(start_barrier);
  #ifdef LATENCY
  uint64_t tagged_jobs = 0;
  #endif

  // /* scheduler routine */
  for(;;){
    if(!busy_coros.empty()){
      coro_info_t* next_coro = busy_coros.front();
      curr_yield = next_coro->yield;

      busy_coros.pop_front();

      #ifdef LATENCY
      if (next_coro->h.tagged){
        if(next_coro->h.num_services == 0){
          next_coro->h.first_served = fenced_rdtscp();
        } else if(next_coro->h.num_services == 1){
          next_coro->h.resumed1 = fenced_rdtscp();
        } else if(next_coro->h.num_services == 2){
          next_coro->h.resumed2 = fenced_rdtscp();
        }
        next_coro->h.num_services++;
      }
      #endif

      (*(next_coro->coro))();
      if(next_coro->jinfo->s == COMPLETED){
        #ifdef LATENCY
        next_coro->h.postfn_completed = fenced_rdtscp();
        #endif
        idle_coros.push_back(next_coro);

        #ifdef LOST_ENQ_TIME
        next_coro->h.failed_enqs = next_coro->jinfo->failed_enq;
        #endif

        while(!c_q->try_enqueue(next_coro->h));
        #ifdef JSQ
        (*p_finished_jobs)++;
        #endif

      } else {
        #ifdef LATENCY
        if(next_coro->h.num_services == 1){
          next_coro->h.prefn_completed = fenced_rdtscp();
        }
        #endif
        busy_coros.push_back(next_coro); // no monitoring set yet -- we can track offloaded_coros in a separate set
      }
    }
    if(idle_coros.empty()){ /* keep cycling through if there are no free idles */
      continue;
    }
    if(q->try_dequeue(h)){
      coro_info_t* next_coro = idle_coros.back();
      next_coro->jinfo->jtype = (job_type_t)h.req_type;
      next_coro->jinfo->args = h.w_args;
      next_coro->h = h;
      idle_coros.pop_back();
      popped++;
      busy_coros.push_front(next_coro);
    }

  }

  free(job_infos);
  free(coros);
  free(c_infos);

  #ifdef LATENCY
  PRINT("[Worker %d] Tagged Jobs: %ld\n", id, tagged_jobs);
  #endif


  // pthread_barrier_wait(args->exit_barrier);
  return NULL;
}

void *worker_rr_sw_fallback(void *arg){
  wrk_args_t *args = static_cast<wrk_args_t*>(arg);
  dispatch_ring_t *q = args->q;
  response_ring_t *c_q = args->c_queue;
  int id = args->id;
  int popped = 0;
  int num_requests = args->num_requests;
  int num_coros = args->num_coros;
  struct numa_mem *nm;
  nm = (struct numa_mem *)calloc(1, sizeof(struct numa_mem));
  int node = args->node;
  int iaa_dev_id = args->iaa_dev_id;
  int dsa_dev_id = args->dsa_dev_id;
  pthread_barrier_t *start_barrier = args->start_barrier;
  hdr h;

  std::vector<coro_info_t*> idle_coros;
  idle_coros.reserve(num_coros);

  coro_t::pull_type *coros
    = (coro_t::pull_type *)malloc(num_coros * sizeof(coro_t::pull_type));
  coro_info_t *c_infos
    = (coro_info_t *)malloc(num_coros * sizeof(coro_info_t));
  job_info_t *job_infos = args->jinfos;

  std::deque<coro_info_t*> busy_coros;


  char *st_pool = args->stack_pool;
  /* initialize coros */
  for(int i = 0; i < num_coros; i++){
    coros[i] =
      coro_t::pull_type(
          SimpleStack(static_cast<char*>(st_pool + (i * STACK_SIZE))),
          boost::bind( coro_rr_sw_fallback, i, &job_infos[i], _1));

    c_infos[i].coro = &coros[i];
    c_infos[i].yield = static_cast<coro_t::push_type*>(coros[i].get());
    c_infos[i].jinfo = &job_infos[i];
    c_infos[i].jinfo->s = INIT;
    c_infos[i].jinfo->failed_enq = 0;
    idle_coros.push_back(&c_infos[i]);
  }

  LOG_PRINT(LOG_DEBUG, "Worker %d allocating IAA %d\n", id, iaa_dev_id);
  thread_wq_alloc(&m_iaa, iaa_dev_id);
  LOG_PRINT(LOG_DEBUG, "Worker %d allocating DSA %d\n", id, dsa_dev_id);
  thread_wq_alloc(&m_dsa, dsa_dev_id);

  #ifdef JSQ
  int *p_finished_jobs = args->p_finished_jobs;
  *p_finished_jobs = 0;
  #endif

  pthread_barrier_wait(start_barrier);
  #ifdef LATENCY
  uint64_t tagged_jobs = 0;
  #endif

  // /* scheduler routine */
  for(;;){
    if(!busy_coros.empty()){
      coro_info_t* next_coro = busy_coros.front();
      curr_yield = next_coro->yield;

      busy_coros.pop_front();

      #ifdef LATENCY
      if (next_coro->h.tagged){
        if(next_coro->h.num_services == 0){
          next_coro->h.first_served = fenced_rdtscp();
        } else if(next_coro->h.num_services == 1){
          next_coro->h.resumed1 = fenced_rdtscp();
        } else if(next_coro->h.num_services == 2){
          next_coro->h.resumed2 = fenced_rdtscp();
        }
        next_coro->h.num_services++;
      }
      #endif

      (*(next_coro->coro))();
      if(next_coro->jinfo->s == COMPLETED){
        #ifdef LATENCY
        next_coro->h.postfn_completed = fenced_rdtscp();
        #endif
        idle_coros.push_back(next_coro);
        while(!c_q->try_enqueue(next_coro->h));
        #ifdef JSQ
        (*p_finished_jobs)++;
        #endif

      } else {
        #ifdef LATENCY
        if(next_coro->h.num_services == 1){
          next_coro->h.prefn_completed = fenced_rdtscp();
        }
        #endif
        busy_coros.push_back(next_coro); // no monitoring set yet -- we can track offloaded_coros in a separate set
      }
    }
    if(idle_coros.empty()){ /* keep cycling through if there are no free idles */
      continue;
    }
    if(q->try_dequeue(h)){
      coro_info_t* next_coro = idle_coros.back();
      next_coro->jinfo->jtype = (job_type_t)h.req_type;
      next_coro->jinfo->args = h.w_args;
      next_coro->h = h;
      idle_coros.pop_back();
      popped++;
      busy_coros.push_front(next_coro);
    }

  }

  free(job_infos);
  free(coros);
  free(c_infos);

  #ifdef LATENCY
  PRINT("[Worker %d] Tagged Jobs: %ld\n", id, tagged_jobs);
  #endif


  // pthread_barrier_wait(args->exit_barrier);
  return NULL;
}

void *worker_ms_cl_opt(void *arg){
  wrk_args_t *args = static_cast<wrk_args_t*>(arg);
  dispatch_ring_t *q = args->q;
  response_ring_t *c_q = args->c_queue;
  int id = args->id;
  int popped = 0;
  int num_requests = args->num_requests;
  int num_coros = args->num_coros;
  struct numa_mem *nm;
  nm = (struct numa_mem *)calloc(1, sizeof(struct numa_mem));
  int node = args->node;
  int iaa_dev_id = args->iaa_dev_id;
  int dsa_dev_id = args->dsa_dev_id;
  pthread_barrier_t *start_barrier = args->start_barrier;
  hdr h;

  std::vector<coro_info_t*> idle_coros;
  coro_t::pull_type *coros
    = (coro_t::pull_type *)malloc(num_coros * sizeof(coro_t::pull_type));
  coro_info_t *c_infos
    = (coro_info_t *)malloc(num_coros * sizeof(coro_info_t));
  job_info_t *job_infos = args->jinfos;
  std::vector<coro_info_t*> coro_set;
  idxd_comp *mon_set = (idxd_comp *)(args->cr_pool);
  std::queue<int> busy_coros;

  coro_set.reserve(num_coros);
  idle_coros.reserve(num_coros);

  char *st_pool = args->stack_pool;
  /* initialize coros */
  for(int i = 0; i < num_coros; i++){
    coros[i] =
      coro_t::pull_type(
          SimpleStack(static_cast<char*>(st_pool + (i * STACK_SIZE))),
          boost::bind( coro, i, &job_infos[i], _1));

    c_infos[i].coro = &coros[i];
    c_infos[i].id = i;
    c_infos[i].yield = static_cast<coro_t::push_type*>(coros[i].get());
    c_infos[i].jinfo = &job_infos[i];
    c_infos[i].jinfo->s = INIT;
    c_infos[i].jinfo->comp = &mon_set[i];
    c_infos[i].jinfo->failed_enq = 0;
    idle_coros[i] = (&c_infos[i]);
    mon_set[i].status = IAX_COMP_NONE;
  }

  LOG_PRINT(LOG_DEBUG, "Worker %d allocating IAA %d\n", id, iaa_dev_id);
  thread_wq_alloc(&m_iaa, iaa_dev_id);
  LOG_PRINT(LOG_DEBUG, "Worker %d allocating DSA %d\n", id, dsa_dev_id);
  thread_wq_alloc(&m_dsa, dsa_dev_id);

  #ifdef JSQ
  int *p_finished_jobs = args->p_finished_jobs;
  *p_finished_jobs = 0;
  #endif

  pthread_barrier_wait(start_barrier);
  #ifdef LATENCY
  uint64_t tagged_jobs = 0;
  #endif
  int next_busy_coro_idx = 0;
  coro_info_t *next_busy_coro = c_infos; /* next busy  coro */
  int next_idle_coro_idx = 0;
  coro_info_t *next_idle_coro = c_infos; /* next idle_coro */
  int next_cr_idx = 0; /* next cr to complete */
  idxd_comp *next_cr = mon_set; /* next completion ring to check */
  for(;;){
    if(next_cr->status == IAX_COMP_SUCCESS){
      if(next_busy_coro->jinfo->s == OFFLOADED){
        (*(next_busy_coro->coro))();
      }
      else if(!next_busy_coro->jinfo->s == COMPLETED){
        PRINT("Error: Coro %d is not in COMPLETED state\n", next_busy_coro->id);
        exit(1);
      }
      while(!c_q->try_enqueue(next_busy_coro->h));
      next_cr->status = IAX_COMP_NONE;
      next_cr_idx = (next_cr_idx + 1) % num_coros;
      next_cr = mon_set + next_cr_idx;
      next_busy_coro_idx = (next_busy_coro_idx + 1) % num_coros;
      next_busy_coro = c_infos + next_busy_coro_idx;
    }

    while(next_busy_coro_idx != ((next_idle_coro_idx + 1) % num_coros)){
      if(!q->try_dequeue(h)){
        break;
      }
      next_idle_coro->jinfo->jtype = (job_type_t)h.req_type;
      next_idle_coro->jinfo->args = h.w_args;
      next_idle_coro->h = h;
      #if defined (COUNT_LOST_ENQS) || defined (LOST_ENQ_TIME)
      next_idle_coro->jinfo->failed_enq = 0;
      #endif
      popped++;
      (*(next_idle_coro->coro))();
      next_idle_coro_idx = (next_idle_coro_idx + 1) % num_coros;
      next_idle_coro = &c_infos[next_idle_coro_idx];
    }
  }

  free(job_infos);
  free(coros);
  free(c_infos);

  #ifdef LATENCY
  PRINT("[Worker %d] Tagged Jobs: %ld\n", id, tagged_jobs);
  #endif
  return NULL;
}

void *worker_ms(void *arg){
  wrk_args_t *args = static_cast<wrk_args_t*>(arg);
  dispatch_ring_t *q = args->q;
  response_ring_t *c_q = args->c_queue;
  int id = args->id;
  int popped = 0;
  int num_requests = args->num_requests;
  int num_coros = args->num_coros;
  struct numa_mem *nm;
  nm = (struct numa_mem *)calloc(1, sizeof(struct numa_mem));
  int node = args->node;
  int iaa_dev_id = args->iaa_dev_id;
  int dsa_dev_id = args->dsa_dev_id;
  pthread_barrier_t *start_barrier = args->start_barrier;
  hdr h;

  std::vector<coro_info_t*> idle_coros;
  coro_t::pull_type *coros
    = (coro_t::pull_type *)malloc(num_coros * sizeof(coro_t::pull_type));
  coro_info_t *c_infos
    = (coro_info_t *)malloc(num_coros * sizeof(coro_info_t));
  job_info_t *job_infos = args->jinfos;
  std::vector<coro_info_t*> coro_set;
  idxd_comp *mon_set = (idxd_comp *)(args->cr_pool);
  std::queue<int> busy_coros;
  coro_info_t *next_coro;

  coro_set.reserve(num_coros);
  idle_coros.reserve(num_coros);

  char *st_pool = args->stack_pool;
  /* initialize coros */
  for(int i = 0; i < num_coros; i++){
    coros[i] =
      coro_t::pull_type(
          SimpleStack(static_cast<char*>(st_pool + (i * STACK_SIZE))),
          boost::bind( coro, i, &job_infos[i], _1));

    c_infos[i].coro = &coros[i];
    c_infos[i].id = i;
    c_infos[i].yield = static_cast<coro_t::push_type*>(coros[i].get());
    c_infos[i].jinfo = &job_infos[i];
    c_infos[i].jinfo->s = INIT;
    c_infos[i].jinfo->comp = &mon_set[i];
    c_infos[i].jinfo->failed_enq = 0;
    coro_set.push_back(&c_infos[i]);
    idle_coros.push_back(&c_infos[i]);
  }

  LOG_PRINT(LOG_DEBUG, "Worker %d allocating IAA %d\n", id, iaa_dev_id);
  thread_wq_alloc(&m_iaa, iaa_dev_id);
  LOG_PRINT(LOG_DEBUG, "Worker %d allocating DSA %d\n", id, dsa_dev_id);
  thread_wq_alloc(&m_dsa, dsa_dev_id);

  #ifdef JSQ
  int *p_finished_jobs = args->p_finished_jobs;
  *p_finished_jobs = 0;
  #endif

  pthread_barrier_wait(start_barrier);
  #ifdef LATENCY
  uint64_t tagged_jobs = 0;
  #endif

  for(;;){
    if(!busy_coros.empty()){
      int idx = busy_coros.front();
      if(mon_set[idx].status != IAX_COMP_NONE){
        next_coro = coro_set[idx];
        (*(next_coro->coro))();
        mon_set[idx].status = IAX_COMP_NONE;
        idle_coros.push_back(next_coro);
        busy_coros.pop();
        #if defined (COUNT_LOST_ENQS) || defined (LOST_ENQ_TIME)
        next_coro->h.failed_enqs = next_coro->jinfo->failed_enq;
        #endif
        while(!c_q->try_enqueue(next_coro->h));
        (*p_finished_jobs)++;
      }
    }

    while(!idle_coros.empty()){
      if(!q->try_dequeue(h)){
        break;
      }
      next_coro = idle_coros.back();
      next_coro->jinfo->jtype = (job_type_t)h.req_type;
      next_coro->jinfo->args = h.w_args;
      next_coro->h = h;
      #if defined (COUNT_LOST_ENQS) || defined (LOST_ENQ_TIME)
      next_coro->jinfo->failed_enq = 0;
      #endif
      idle_coros.pop_back();
      popped++;
      (*(next_coro->coro))(); // only execute one new request per iteration
      busy_coros.push(next_coro->id);
    }
  }

  free(job_infos);
  free(coros);
  free(c_infos);

  #ifdef LATENCY
  PRINT("[Worker %d] Tagged Jobs: %ld\n", id, tagged_jobs);
  #endif
  return NULL;
}

void *worker_noacc(void *arg){
  wrk_args_t *args = static_cast<wrk_args_t*>(arg);
  dispatch_ring_t *q = args->q;
  response_ring_t *c_q = args->c_queue;
  int id = args->id;
  int popped = 0;
  int num_requests = args->num_requests;
  int num_coros = args->num_coros;
  struct numa_mem *nm;
  nm = (struct numa_mem *)calloc(1, sizeof(struct numa_mem));
  int node = args->node;
  int iaa_dev_id = args->iaa_dev_id;
  int dsa_dev_id = args->dsa_dev_id;
  pthread_barrier_t *start_barrier = args->start_barrier;
  hdr h;

  std::vector<coro_info_t*> idle_coros;
  coro_t::pull_type *coros
    = (coro_t::pull_type *)malloc(num_coros * sizeof(coro_t::pull_type));
  coro_info_t *c_infos
    = (coro_info_t *)malloc(num_coros * sizeof(coro_info_t));
  job_info_t *job_infos = args->jinfos;
  std::vector<coro_info_t*> coro_set;
  idxd_comp *mon_set = (idxd_comp *)(args->cr_pool);
  std::queue<int> busy_coros;
  coro_info_t *next_coro;

  coro_set.reserve(num_coros);
  idle_coros.reserve(num_coros);

  char *st_pool = args->stack_pool;
  /* initialize coros */
  for(int i = 0; i < num_coros; i++){
    coros[i] =
      coro_t::pull_type(
          SimpleStack(static_cast<char*>(st_pool + (i * STACK_SIZE))),
          boost::bind( coro_noacc, i, &job_infos[i], _1));

    c_infos[i].coro = &coros[i];
    c_infos[i].id = i;
    c_infos[i].yield = static_cast<coro_t::push_type*>(coros[i].get());
    c_infos[i].jinfo = &job_infos[i];
    c_infos[i].jinfo->s = INIT;
    c_infos[i].jinfo->comp = &mon_set[i];
    c_infos[i].jinfo->failed_enq = 0;
    coro_set.push_back(&c_infos[i]);
    idle_coros.push_back(&c_infos[i]);
  }

  LOG_PRINT(LOG_DEBUG, "Worker %d allocating IAA %d\n", id, iaa_dev_id);
  thread_wq_alloc(&m_iaa, iaa_dev_id);
  LOG_PRINT(LOG_DEBUG, "Worker %d allocating DSA %d\n", id, dsa_dev_id);
  thread_wq_alloc(&m_dsa, dsa_dev_id);

  #ifdef JSQ
  int *p_finished_jobs = args->p_finished_jobs;
  *p_finished_jobs = 0;
  #endif

  pthread_barrier_wait(start_barrier);
  #ifdef LATENCY
  uint64_t tagged_jobs = 0;
  #endif
  next_coro = idle_coros.back();
  for(;;){
      while(!q->try_dequeue(h));
      next_coro->jinfo->jtype = (job_type_t)h.req_type;
      next_coro->jinfo->args = h.w_args;
      next_coro->h = h;
      #if defined (COUNT_LOST_ENQS) || defined (LOST_ENQ_TIME)
      next_coro->jinfo->failed_enq = 0;
      #endif
      (*(next_coro->coro))();
      while(!c_q->try_enqueue(next_coro->h));
      (*p_finished_jobs)++;
  }

  free(coros);
  free(c_infos);

  #ifdef LATENCY
  PRINT("[Worker %d] Tagged Jobs: %ld\n", id, tagged_jobs);
  #endif
  return NULL;
}

void *worker_blocking(void *arg){
  wrk_args_t *args = static_cast<wrk_args_t*>(arg);
  dispatch_ring_t *q = args->q;
  response_ring_t *c_q = args->c_queue;
  int id = args->id;
  int popped = 0;
  int num_requests = args->num_requests;
  int num_coros = args->num_coros;
  struct numa_mem *nm;
  nm = (struct numa_mem *)calloc(1, sizeof(struct numa_mem));
  int node = args->node;
  int iaa_dev_id = args->iaa_dev_id;
  int dsa_dev_id = args->dsa_dev_id;
  pthread_barrier_t *start_barrier = args->start_barrier;
  hdr h;

  std::vector<coro_info_t*> idle_coros;
  coro_t::pull_type *coros
    = (coro_t::pull_type *)malloc(num_coros * sizeof(coro_t::pull_type));
  coro_info_t *c_infos
    = (coro_info_t *)malloc(num_coros * sizeof(coro_info_t));
  job_info_t *job_infos = args->jinfos;
  std::vector<coro_info_t*> coro_set;
  idxd_comp *mon_set = (idxd_comp *)(args->cr_pool);
  std::queue<int> busy_coros;
  coro_info_t *next_coro;

  coro_set.reserve(num_coros);
  idle_coros.reserve(num_coros);

  char *st_pool = args->stack_pool;
  /* initialize coros */
  for(int i = 0; i < num_coros; i++){
    coros[i] =
      coro_t::pull_type(
          SimpleStack(static_cast<char*>(st_pool + (i * STACK_SIZE))),
          boost::bind( coro_blocking, i, &job_infos[i], _1));

    c_infos[i].coro = &coros[i];
    c_infos[i].id = i;
    c_infos[i].yield = static_cast<coro_t::push_type*>(coros[i].get());
    c_infos[i].jinfo = &job_infos[i];
    c_infos[i].jinfo->s = INIT;
    c_infos[i].jinfo->comp = &mon_set[i];
    c_infos[i].jinfo->failed_enq = 0;
    coro_set.push_back(&c_infos[i]);
    idle_coros.push_back(&c_infos[i]);
  }

  LOG_PRINT(LOG_DEBUG, "Worker %d allocating IAA %d\n", id, iaa_dev_id);
  thread_wq_alloc(&m_iaa, iaa_dev_id);
  LOG_PRINT(LOG_DEBUG, "Worker %d allocating DSA %d\n", id, dsa_dev_id);
  thread_wq_alloc(&m_dsa, dsa_dev_id);

  #ifdef JSQ
  int *p_finished_jobs = args->p_finished_jobs;
  *p_finished_jobs = 0;
  #endif

  pthread_barrier_wait(start_barrier);
  #ifdef LATENCY
  uint64_t tagged_jobs = 0;
  #endif
  next_coro = idle_coros.back();
  for(;;){
      while(!q->try_dequeue(h));
      next_coro->jinfo->jtype = (job_type_t)h.req_type;
      next_coro->jinfo->args = h.w_args;
      next_coro->h = h;
      #if defined (COUNT_LOST_ENQS) || defined (LOST_ENQ_TIME)
      next_coro->jinfo->failed_enq = 0;
      #endif
      (*(next_coro->coro))();
      #if defined (COUNT_LOST_ENQS) || defined (LOST_ENQ_TIME)
      next_coro->h.failed_enqs = next_coro->jinfo->failed_enq;
      #endif

      while(!c_q->try_enqueue(next_coro->h));
      (*p_finished_jobs)++;
  }

  free(coros);
  free(c_infos);

  #ifdef LATENCY
  PRINT("[Worker %d] Tagged Jobs: %ld\n", id, tagged_jobs);
  #endif
  return NULL;
}

void *monitor(void *arg){
  /*
  algo: loop through all response rings dequeueing and counting
  once the warmups have been accounted for, start timing
  once last expected request has been collected, stop timing
  -- check for tagged jobs as we collect
  */
  monitor_args_t *args = static_cast<monitor_args_t*>(arg);
  int num_pulled = 0;// = args->num_requests;
  int expected = args->num_requests;
  int num_workers = args->num_workers;
  response_ring_t **c_queues = args->c_queues;
  int warmup = expected / 10;
  uint64_t start, end;
  int dq_idx = 0;
  hdr h;
  uint64_t *cs_per_wrkr = (uint64_t *)malloc(num_workers * sizeof(uint64_t));
  for(int i=0; i<num_workers; i++){
    cs_per_wrkr[i] = 0;
  }
  #ifdef LATENCY
  int tagged_jobs = 0;
  hdr *hs =
    (hdr *)malloc(expected * sizeof(hdr));
  #endif

  while(num_requests.load() == 0){
    _mm_pause();
  }
  expected = num_requests.load();
  pthread_barrier_wait(args->start_barrier);
  #ifdef THROUGHPUT
  while(num_pulled < warmup){
    dq_idx = (dq_idx + 1) % num_workers;
    if(c_queues[dq_idx]->try_dequeue(h)){
      #ifdef LATENCY
      if(h.tagged){
        uint64_t compl_time = fenced_rdtscp();
        h.completed = compl_time;
        hs[tagged_jobs ] = h;
        tagged_jobs++;
      }
      #endif
      num_pulled++;
      cs_per_wrkr[dq_idx]++;
    }

  }
  start = fenced_rdtscp();
  #endif

  while(num_pulled < expected){
    dq_idx = (dq_idx + 1) % num_workers;
    if(c_queues[dq_idx]->try_dequeue(h)){ /* if tagged, take time stamp and add to tagged timer table */
      #ifdef LATENCY
      if(h.tagged){
        uint64_t compl_time = fenced_rdtscp();
        h.completed = compl_time;
        hs[tagged_jobs ] = h;
        tagged_jobs++;
      }
      #endif

      num_pulled++;
      cs_per_wrkr[dq_idx]++;
    }
  }
  #ifdef THROUGHPUT
  end = fenced_rdtscp();
  #endif

  #ifdef LATENCY
  while(workload_start_cycle == 0);
  uint64_t arrival_cycle_0 = workload_start_cycle.load();
  PRINT("[Monitor %d] Workload Start Cycle: %ld\n", args->id, workload_start_cycle.load());
  #endif

  uint64_t freqKHz = 2400000;
  PRINT("[Monitor %d ] Response Rate (MRPS): %f\n", args->id, (double)(expected-warmup) / ((double)(end - start) / 2100));
  #ifdef LATENCY
  PRINT("[Monitor %d] Tagged Jobs: %d\n", args->id, tagged_jobs);
  /* Avg, Median, 99p of tagged jobs */
  std::vector<uint64_t> differences;
  std::vector<uint64_t> t_in_wq;
  std::vector<uint64_t> t_in_c1;
  std::vector<uint64_t> t_in_resumq1;
  std::vector<uint64_t> t_in_resumq2;
  std::vector<uint64_t> t_in_c2;
  std::vector<uint64_t> num_services;
  std::vector<uint64_t> t_in_dispatchq;
  #if defined(LOST_ENQ_TIME) || defined(COUNT_LOST_ENQS)
  std::vector<uint64_t> lost_enq_times;
  std::vector<double> frac_spent_on_enqs;
  #endif
  double median;
  int num_unknown_post_services = 0;
  for(int i=0; i<tagged_jobs; i++){
    if(hs[i].id != 0){ /* TODO: currently some requests are making it here with unmarked headers -- just discard these for now */
    if(hs[i].num_services > 0){
      t_in_wq.push_back(hs[i].first_served - hs[i].dispatched);
    }
    if(hs[i].num_services > 1){
      t_in_resumq1.push_back(hs[i].resumed1 - hs[i].first_served);
    }
    if(hs[i].num_services > 2){
      t_in_resumq2.push_back(hs[i].resumed2 - hs[i].resumed1);
    }
    if(hs[i].num_services == 2){
      t_in_c2.push_back(hs[i].postfn_completed - hs[i].resumed1);
    } else if (hs[i].num_services == 3){
      t_in_c2.push_back(hs[i].postfn_completed - hs[i].resumed2);
    } else {
      num_unknown_post_services++;
    }
    if (hs[i].prefn_completed > hs[i].first_served && hs[i].prefn_completed != 0 && hs[i].first_served != 0) {
      t_in_c1.push_back(hs[i].prefn_completed - hs[i].first_served);
    }
    uint64_t arrival_cycle = arrival_cycle_0 + hs[i].arrival;
    t_in_dispatchq.push_back(hs[i].dispatched - arrival_cycle);
    num_services.push_back(hs[i].num_services);
    differences.push_back(hs[i].completed - arrival_cycle);

    #if defined(LOST_ENQ_TIME) || defined(COUNT_LOST_ENQS)
    lost_enq_times.push_back(hs[i].failed_enqs);
    frac_spent_on_enqs.push_back((double)hs[i].failed_enqs / (hs[i].completed - arrival_cycle));
    #endif

      LOG_PRINT(LOG_DEBUG,"[Monitor %d] Id:%ld, "
        "Arrival: %ld, "
        "Injected: %ld, "
        "Dispatched: %ld, "
        "First Served: %ld, "
        "Resumed1: %ld, "
        "Resumed2: %ld, "
        "Num Services: %d, "
        "Completed: %ld, "
        "Latency: %ld\n", args->id, hs[i].id,
        hs[i].arrival,
        hs[i].injected,
        hs[i].dispatched, hs[i].first_served, hs[i].resumed1, hs[i].resumed2,
        hs[i].num_services, hs[i].completed, hs[i].completed - hs[i].injected);
    }
  }

  std::sort(differences.begin(), differences.end());
  median = differences[differences.size() / 2];
  PRINT("[Monitor %d] Median Latency: %f\n", args->id, median);

  size_t idx_99 = static_cast<size_t>(0.99 * differences.size());
  double p99 = differences[idx_99];
  PRINT("[Monitor %d] 99th Percentile Latency: %f\n", args->id, p99);

  double avg = std::accumulate(differences.begin(), differences.end(), 0.0) / differences.size();
  PRINT("[Monitor %d] Average Latency: %f\n", args->id, avg);

  auto calculate_average = [](const std::vector<uint64_t>& times) -> double {
    if (times.empty()) return 0.0;
    return std::accumulate(times.begin(), times.end(), 0.0) / times.size();
  };
  auto calculate_percentile = [](const std::vector<uint64_t>& times, double percentile) -> double {
    if (times.empty()) return 0.0;
    size_t idx = static_cast<size_t>(percentile * times.size());
    return times[idx];
  };

  auto calculate_average_double = [](const std::vector<double>& values) -> double {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  };

  auto calculate_percentile_double = [](const std::vector<double>& values, double percentile) -> double {
    if (values.empty()) return 0.0;
    size_t idx = static_cast<size_t>(percentile * values.size());
    return values[idx];
  };

  std::sort(t_in_wq.begin(), t_in_wq.end());
  std::sort(t_in_resumq1.begin(), t_in_resumq1.end());
  std::sort(t_in_resumq2.begin(), t_in_resumq2.end());

  double avg_time_in_wq = calculate_average(t_in_wq);
  double median_time_in_wq = calculate_percentile(t_in_wq, 0.5);
  double p99_time_in_wq = calculate_percentile(t_in_wq, 0.99);

  double avg_time_in_resumq1 = calculate_average(t_in_resumq1);
  double median_time_in_resumq1 = calculate_percentile(t_in_resumq1, 0.5);
  double p99_time_in_resumq1 = calculate_percentile(t_in_resumq1, 0.99);

  double avg_time_in_resumq2 = calculate_average(t_in_resumq2);
  double median_time_in_resumq2 = calculate_percentile(t_in_resumq2, 0.5);
  double p99_time_in_resumq2 = calculate_percentile(t_in_resumq2, 0.99);


  std::sort(num_services.begin(), num_services.end());

  double avg_num_services = calculate_average(num_services);
  double median_num_services = calculate_percentile(num_services, 0.5);
  double p99_num_services = calculate_percentile(num_services, 0.99);

  PRINT("[Monitor %d] Average Number of Services: %f\n", args->id, avg_num_services);
  PRINT("[Monitor %d] Median Number of Services: %f\n", args->id, median_num_services);
  PRINT("[Monitor %d] 99th Percentile Number of Services: %f\n", args->id, p99_num_services);


  std::sort(t_in_dispatchq.begin(), t_in_dispatchq.end());

  double avg_time_in_dispatchq = calculate_average(t_in_dispatchq);
  double median_time_in_dispatchq = calculate_percentile(t_in_dispatchq, 0.5);
  double p99_time_in_dispatchq = calculate_percentile(t_in_dispatchq, 0.99);

  PRINT("[Monitor %d] Average Time in Dispatch Queue: %f\n", args->id, avg_time_in_dispatchq);
  PRINT("[Monitor %d] Median Time in Dispatch Queue: %f\n", args->id, median_time_in_dispatchq);
  PRINT("[Monitor %d] 99th Percentile Time in Dispatch Queue: %f\n", args->id, p99_time_in_dispatchq);

  PRINT("[Monitor %d] Average Time in Worker Queue: %f\n", args->id, avg_time_in_wq);
  PRINT("[Monitor %d] Median Time in Worker Queue: %f\n", args->id, median_time_in_wq);
  PRINT("[Monitor %d] 99th Percentile Time in Worker Queue: %f\n", args->id, p99_time_in_wq);

  std::sort(t_in_c1.begin(), t_in_c1.end());

  double avg_time_in_c1 = calculate_average(t_in_c1);
  double median_time_in_c1 = calculate_percentile(t_in_c1, 0.5);
  double p99_time_in_c1 = calculate_percentile(t_in_c1, 0.99);

  PRINT("[Monitor %d] Average Time in Core 1st Service: %f\n", args->id, avg_time_in_c1);
  PRINT("[Monitor %d] Median Time in Core 1st Service: %f\n", args->id, median_time_in_c1);
  PRINT("[Monitor %d] 99th Percentile Time in Core 1st Service: %f\n", args->id, p99_time_in_c1);

  PRINT("[Monitor %d] Average Time in Resumption Queue 1: %f\n", args->id, avg_time_in_resumq1);
  PRINT("[Monitor %d] Median Time in Resumption Queue 1: %f\n", args->id, median_time_in_resumq1);
  PRINT("[Monitor %d] 99th Percentile Time in Resumption Queue 1: %f\n", args->id, p99_time_in_resumq1);

  std::sort(t_in_c2.begin(), t_in_c2.end());

  double avg_time_in_c2 = calculate_average(t_in_c2);
  double median_time_in_c2 = calculate_percentile(t_in_c2, 0.5);
  double p99_time_in_c2 = calculate_percentile(t_in_c2, 0.99);

  PRINT("[Monitor %d] Average Time in Resumption Queue 2: %f\n", args->id, avg_time_in_resumq2);
  PRINT("[Monitor %d] Median Time in Resumption Queue 2: %f\n", args->id, median_time_in_resumq2);
  PRINT("[Monitor %d] 99th Percentile Time in Resumption Queue 2: %f\n", args->id, p99_time_in_resumq2);


  PRINT("[Monitor %d] Average Time in Core 2nd Service: %f\n", args->id, avg_time_in_c2);
  PRINT("[Monitor %d] Median Time in Core 2nd Service: %f\n", args->id, median_time_in_c2);
  PRINT("[Monitor %d] 99th Percentile Time in Core 2nd Service: %f\n", args->id, p99_time_in_c2);

  #if defined(LOST_ENQ_TIME) || defined(COUNT_LOST_ENQS)
  std::sort(lost_enq_times.begin(), lost_enq_times.end());

  double avg_lost_enq_time = calculate_average(lost_enq_times);
  double median_lost_enq_time = calculate_percentile(lost_enq_times, 0.5);
  double p99_lost_enq_time = calculate_percentile(lost_enq_times, 0.99);

  PRINT("[Monitor %d] Average Lost Enqueue Time: %f\n", args->id, avg_lost_enq_time);
  PRINT("[Monitor %d] Median Lost Enqueue Time: %f\n", args->id, median_lost_enq_time);
  PRINT("[Monitor %d] 99th Percentile Lost Enqueue Time: %f\n", args->id, p99_lost_enq_time);

  double avg_frac_spent_on_enqs = calculate_average_double(frac_spent_on_enqs);
  PRINT("[Monitor %d] Average Fraction of Time Spent on Enqueues: %f\n", args->id, avg_frac_spent_on_enqs);
  #endif


  std::vector<double> slowdowns;
  for (int i = 0; i < tagged_jobs; i++) {
    if (hs[i].unloaded > 0) {
      double slowdown = differences[i]/ (double)hs[i].unloaded;
      slowdowns.push_back(slowdown);
    }
  }

  std::sort(slowdowns.begin(), slowdowns.end());

  double avg_slowdown = calculate_average_double(slowdowns);
  double p99_slowdown = calculate_percentile_double(slowdowns, 0.999);

  PRINT("[Monitor %d] Average Slowdown: %f\n", args->id, avg_slowdown);
  PRINT("[Monitor %d] 99.9th Percentile Slowdown: %f\n", args->id, p99_slowdown);
  double median_slowdown = calculate_percentile_double(slowdowns, 0.5);
  PRINT("[Monitor %d] Median Slowdown: %f\n", args->id, median_slowdown);

  PRINT("[Monitor %d] Load Distribution:\n", args->id);
  for (int i = 0; i < num_workers; i++) {
    PRINT("W%d: %lu ", i, cs_per_wrkr[i]);
  }
  PRINT("\n");
  #endif

  free(cs_per_wrkr);

  PRINT("Config: ");
  PRINT("requests: %d ", expected);
  PRINT("workers: %d ", num_workers);
  PRINT("coros_per_worker: %d ", args->coros_per_worker);
  PRINT("IAAs: %d ", args->num_iaas);
  PRINT("DSAs: %d ", args->num_dsas);
  PRINT("Inqueue_size: %d ", args->inqueue_size_elems);
  PRINT("Dispatch_queue_size: %d\n", args->dispatch_queue_num_elems);

  pthread_barrier_wait(args->exit_barrier);
  // we know everything is done -- exit
  exit(0);
  return NULL;
}

int gLogLevel = LOG_DEBUG;
bool gDebugParam = false;
int main(int argc, char **argv){
  int num_iaas = 2;
  int num_dsas = 2;
  int num_reqs = 15000000;
  int num_workers = 4;
  int coros_per_worker = 4;
  int inqueue_size_bytes = 4096;
  int inqueue_size_elems;
  int dispatch_queue_size_bytes = 4096;
  int dispatch_queue_num_elems;
  int completion_queue_size_bytes = 4096;
  int completion_queue_num_elems;
  int server_start_core = 26;
  int server_node = 1;
  int wrkload_gen_core = 1;
  int monitor_core = 2;
  double load = 1.0;
  double peak = 1024.0;
  double peak2 = 1024.0;
  bool unloaded = false;
  uint64_t num_ddh_requests = 0;
  uint64_t num_dmdp_requests = 0;
  uint64_t num_dg_requests = 0;
  uint64_t num_mg_requests = 0;
  uint64_t num_ufh_requests = 0;
  uint64_t num_mmp_requests = 0;
  dist_t dist_type = EXPONENTIAL;
  worker_type_t worker_type = WORKER_MS_CL_OPT;

  pthread_barrier_t *exit_barrier = (pthread_barrier_t *)malloc(sizeof(pthread_barrier_t));
  pthread_barrier_t *start_barrier = (pthread_barrier_t *)malloc(sizeof(pthread_barrier_t));

  int opt;
  while((opt = getopt(argc, argv, "n:c:q:d:i:w:l:s:y:u:p:z:x:v:b:m:k:j:h:g:f:a:")) != -1){
    switch(opt){
      case 's':
        server_start_core = atoi(optarg);
        break;
      case 'y':
        server_node = atoi(optarg);
        break;
      case 'u':
        wrkload_gen_core = atoi(optarg);
        break;
      case 'n':
        num_reqs = atoi(optarg);
        break;
      case 'c':
        coros_per_worker = atoi(optarg);
        break;
      case 'q':
        inqueue_size_bytes = atoi(optarg);
        break;
      case 'd':
        dispatch_queue_size_bytes = atoi(optarg);
        completion_queue_size_bytes = atoi(optarg);
        break;
      case 'i':
        num_iaas = atoi(optarg);
        break;
      case 'w':
        num_workers = atoi(optarg);
        break;
      case 'l':
        gLogLevel = atoi(optarg);
        break;
      case 'p':
        load = atof(optarg);
        break;
      case 'z':
        num_ddh_requests = atoll(optarg);
        break;
      case 'x':
        num_dmdp_requests = atoll(optarg);
        break;
      case 'v':
        dist_type = (dist_t)atoi(optarg);
        break;
      case 'h':
        num_dg_requests = atoll(optarg);
        break;
      case 'b':
        peak = atof(optarg);
        break;
      case 'k':
        peak2 = atof(optarg);
        break;
      case 'm':
        worker_type = (worker_type_t)atoi(optarg);
        break;
      case 'j':
        unloaded = true;
        break;
      case 'g':
        num_mg_requests = atoll(optarg);
        break;
      case 'f':
        num_ufh_requests = atoll(optarg);
        break;
      case 'a':
        num_mmp_requests = atoll(optarg);
        break;
      default:
        printf("Usage: %s -n <num_reqs> -w <num_workers>\n", argv[0]);
        return -1;
    }
  }

  pthread_t dispatcher_thread;
  pthread_t wrlkd_thread;
  pthread_t *worker_threads = (pthread_t *)malloc(num_workers * sizeof(pthread_t));
  pthread_t monitor_thread ; //= (pthread_t *)malloc(num_workers * sizeof(pthread_t));
  wrkld_gen_args_t *wrkld_args = (wrkld_gen_args_t *)malloc(sizeof(wrkld_gen_args_t));
  inqueue_size_elems= inqueue_size_bytes / sizeof(hdr);
  wrkload_ring_t *wrkload_q = new wrkload_ring_t(inqueue_size_elems);
  dispatch_ring_t **dispatch_q = (dispatch_ring_t **)malloc(num_workers * sizeof(dispatch_ring_t *));
  dispatch_args_t *dispatch_args = (dispatch_args_t *)malloc(sizeof(dispatch_args_t));
  monitor_args_t *monitor_args = (monitor_args_t *)malloc(sizeof(monitor_args_t ));
  response_ring_t **c_queues = (response_ring_t **)malloc(num_workers * sizeof(response_ring_t *));
  #ifdef JSQ
  int **pp_finished_jobs = (int **)malloc(num_workers * sizeof(int *));
  for(int i=0; i<num_workers; i++){
    pp_finished_jobs[i] = (int *)aligned_alloc(CACHE_LINE_SIZE, CACHE_LINE_SIZE);
    *(pp_finished_jobs[i]) = 0;
  }
  #endif

  struct numa_mem *stack_nm = (struct numa_mem *)calloc(1, sizeof(struct numa_mem));
  reserve_numa_mem(stack_nm, num_workers * coros_per_worker * STACK_SIZE, server_node);
  idxd_comp *cr_pool =
    (idxd_comp *)alloc_numa_offset(stack_nm, num_workers * coros_per_worker * sizeof(idxd_comp), 0);
  job_info_t *jinfos = (job_info_t *)malloc(num_workers * coros_per_worker * sizeof(job_info_t));
  int rc = alloc_numa_mem(stack_nm, PAGE_SIZE, server_node);
  if(rc != 0){
    LOG_PRINT(LOG_ERR, "Error allocating stack memory\n");
    return -1;
  }
  add_base_addr(stack_nm, (void **)&cr_pool);

  PRINT("Request Size: %ld\n", sizeof(hdr));

  pthread_barrier_init(exit_barrier, NULL, 3);
  pthread_barrier_init(start_barrier, NULL, num_workers + 3);

  dispatch_args->wqs = dispatch_q;
  dispatch_args->inq = wrkload_q;
  dispatch_args->num_workers = num_workers;
  dispatch_args->num_requests = num_reqs;
  dispatch_args->exit_barrier = exit_barrier;
  dispatch_args->num_requests = num_reqs;
  dispatch_args->start_barrier = start_barrier;
  #ifdef JSQ
  dispatch_args->pp_finished_jobs = pp_finished_jobs;
  #endif

  wrkld_args->q = wrkload_q;
  wrkld_args->num_reqs = num_reqs;
  wrkld_args->pushed = (int *)malloc(sizeof(int));
  wrkld_args->start = (uint64_t *)malloc(sizeof(uint64_t));
  wrkld_args->end = (uint64_t *)malloc(sizeof(uint64_t));
  wrkld_args->server_node = 1;
  wrkld_args->exit_barrier = exit_barrier;
  wrkld_args->start_barrier = start_barrier;
  wrkld_args->num_ddh_reqs = num_ddh_requests;
  wrkld_args->num_dmdp_reqs = num_dmdp_requests;
  wrkld_args->num_dg_reqs = num_dg_requests;
  wrkld_args->num_mg_reqs = num_mg_requests;
  wrkld_args->num_ufh_reqs = num_ufh_requests;
  wrkld_args->num_mmp_reqs = num_mmp_requests;
  wrkld_args->dist_type = dist_type;
  wrkld_args->peak = peak;
  wrkld_args->peak2 = peak2;
  wrkld_args->unloaded = unloaded;
  // wrkld_args->stack_pool = (char *)stack_nm->base_addr;
  #ifdef POISSON
  wrkld_args->load = load;
  #endif
  dispatch_queue_num_elems = dispatch_queue_size_bytes / sizeof(hdr);
  completion_queue_num_elems = completion_queue_size_bytes / sizeof(hdr);
  for(int i=0; i<num_workers; i++){
    dispatch_q[i] = new dispatch_ring_t(dispatch_queue_num_elems);
    c_queues[i] = new response_ring_t(completion_queue_num_elems);
  }


  create_thread_pinned(&wrlkd_thread, workload_gen, static_cast<void *>(wrkld_args), wrkload_gen_core);
  // if(unloaded){
  //   pthread_join(wrlkd_thread, nullptr);
  // }

  if ( num_reqs % num_workers != 0){
    printf("num_reqs must be divisible by num_workers\n");
    return -1;
  }
  int worker_start_core = server_start_core + 1;
  wrk_args_t **wrk_args = (wrk_args_t **)malloc(num_workers * sizeof(wrk_args_t *));
  for(int i = 0; i < num_workers; i++){
    wrk_args[i] = (wrk_args_t *)malloc(sizeof(wrk_args_t));
    wrk_args[i]->id = i;
    wrk_args[i]->q = dispatch_q[i];
    wrk_args[i]->num_requests = num_reqs / num_workers;
    wrk_args[i]->num_coros = coros_per_worker;
    wrk_args[i]->c_queue = c_queues[i];
    wrk_args[i]->iaa_dev_id = IAA_START + ((i % num_iaas) * IDXD_DEV_STEP);
    wrk_args[i]->dsa_dev_id = DSA_START + ((i % num_dsas) * IDXD_DEV_STEP);
    wrk_args[i]->node = server_node;
    wrk_args[i]->start_barrier = start_barrier;
    wrk_args[i]->exit_barrier = exit_barrier;
    wrk_args[i]->stack_pool = (char *)stack_nm->base_addr + (i * coros_per_worker * STACK_SIZE);
    wrk_args[i]->cr_pool =  cr_pool + (i * coros_per_worker);
    wrk_args[i]->jinfos = jinfos + (i * coros_per_worker);
    #ifdef JSQ
    wrk_args[i]->p_finished_jobs = pp_finished_jobs[i];
    #endif
    switch(worker_type){
      case WORKER_MS:
        PRINT("Creating worker type MS\n");
        create_thread_pinned(&worker_threads[i], worker_ms, static_cast<void*>(wrk_args[i]), worker_start_core + i);
        break;
      case WORKER_RR:
        PRINT("Creating worker type RR\n");
        create_thread_pinned(&worker_threads[i], worker_rr, static_cast<void*>(wrk_args[i]), worker_start_core + i);
        break;
      case WORKER_MS_CL_OPT:
        PRINT("Creating worker type MS_CL_OPT\n");
        create_thread_pinned(&worker_threads[i], worker_ms_cl_opt, static_cast<void*>(wrk_args[i]), worker_start_core + i);
        break;
      case WORKER_NOACC:
        PRINT("Creating worker type NOACC\n");
        create_thread_pinned(&worker_threads[i], worker_noacc, static_cast<void*>(wrk_args[i]), worker_start_core + i);
        break;
      case WORKER_BLOCKING:
        PRINT("Creating worker type BLOCKING\n");
        create_thread_pinned(&worker_threads[i], worker_blocking, static_cast<void*>(wrk_args[i]), worker_start_core + i);
        break;
      case WORKER_RR_SW_FALLBACK:
        PRINT("Creating worker type RR_SW_FALLBACK\n");
        create_thread_pinned(&worker_threads[i], worker_rr_sw_fallback, static_cast<void*>(wrk_args[i]), worker_start_core + i);
        break;
      default:
        PRINT("Creating worker type MS\n");
        create_thread_pinned(&worker_threads[i], worker_ms, static_cast<void*>(wrk_args[i]), worker_start_core + i);
        break;
    }
  }

    monitor_args = (monitor_args_t *)malloc(sizeof(monitor_args_t));
    monitor_args->start_barrier = start_barrier;
    monitor_args->exit_barrier = exit_barrier;
    monitor_args->c_queues = c_queues;
    monitor_args->num_requests = num_reqs;
    monitor_args->num_workers = num_workers;
    monitor_args->coros_per_worker = coros_per_worker;
    monitor_args->num_iaas = num_iaas;
    monitor_args->num_dsas = num_dsas;
    monitor_args->inqueue_size_elems = inqueue_size_elems;
    monitor_args->dispatch_queue_num_elems = dispatch_queue_num_elems;
    monitor_args->jinfos = jinfos;

    monitor_args->id = num_workers;
    create_thread_pinned(&monitor_thread, monitor, static_cast<void*>(monitor_args), monitor_core);

  create_thread_pinned(&dispatcher_thread, dispatcher, static_cast<void *>(dispatch_args), server_start_core);


  pthread_join(dispatcher_thread, nullptr);
  for(int i = 0; i < num_workers; i++){
    pthread_join(worker_threads[i], nullptr);
  }
  // for(int i = 0; i < num_workers; i++){
  pthread_join(monitor_thread, nullptr);
  // }
  pthread_join(wrlkd_thread, nullptr);

  free(wrkld_args->pushed);
  free(wrkld_args->start);
  free(wrkld_args->end);
  free(wrkld_args);
  free(dispatch_args);
  free(wrkload_q);
  for(int i=0; i<num_workers; i++){
    delete dispatch_q[i];
  }
  free(dispatch_q);
  for(int i=0; i<num_workers; i++){
    free(wrk_args[i]);
  }
  free(wrk_args);
  for(int i=0; i<num_workers; i++){
    delete c_queues[i];
  }
  // for(int i=0; i<num_workers; i++){
  // }
  free(c_queues);
  free(monitor_args);
  // free(monitor_threads);
  free(worker_threads);


  #ifdef DEBUG
  PRINT("Completed %d requests\n", num_completed.load());
  assert(num_completed.load() == num_reqs);
  #endif


  return 0;
}