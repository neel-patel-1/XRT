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

#include <list>

#define DSA_START 4
#define IAA_START 5
#define IDXD_DEV_STEP 2
#define DISPATCH_RING_SIZE 4096
#define INQ_SIZE 4096
#define NUM_APPS 2
#define MAX_SER_OVERHEAD_BYTES 128
#define STACK_SIZE (128 * 1024)
#define likely(x)       __builtin_expect((x),1)
#define JSBQ_LEN 2

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

typedef struct completion_record idxd_comp;
typedef struct hw_desc idxd_desc;

typedef enum job_type {
  DESER_DECOMP_HASH,
  DECRYPT_MEMCPY_DP
} job_type_t;
typedef enum status {
  INIT,
  OFFLOADED,
  OFFLOAD_STARTED,
  PREEMPTED,
  COMPLETED
} status;
typedef struct job_info {
    job_type_t jtype;
    status s;
    void *args;
} job_info_t;

typedef struct coro_info {
  coro_t::pull_type *coro;
  job_info_t *jinfo;
  idxd_comp *comp;
  #ifdef LATENCY
  uint64_t id;
  bool tagged;
  uint64_t arrival;
  uint64_t dispatched1;
  uint64_t first_served;
  uint64_t dispatched2;
  uint64_t served_again;
  uint64_t completed;
  #endif
} coro_info_t;
typedef struct req_hdr {
  uint64_t id;
  job_type req_type;
  void *w_args;
  coro_info_t *c_info;
  idxd_comp *comp;
  #ifdef LATENCY
  bool tagged;
  uint64_t arrival;
  uint64_t dispatched1;
  uint64_t first_served;
  uint64_t dispatched2;
  uint64_t served_again;
  uint64_t completed;
  #endif
} hdr; // total no latency -- 20


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
  char *stack_pool;
  int num_requests;
  int num_workers;
  int num_coros;
  pthread_barrier_t *start_barrier;
  pthread_barrier_t *exit_barrier;
  #ifdef JSQ
  int **pp_finished_jobs;
  #endif
  int **pp_completed;
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
  int *p_completed;
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
  int inqueue_size_elems;
  int dispatch_queue_num_elems;
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

__thread struct acctest_context *m_iaa = NULL;

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
    LOG_PRINT(LOG_ERR, "Not enough space to serialize\n");
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

#ifdef POISSON
void gen_interarrival_poisson(double lambda, uint64_t *offsets, int num_reqs, uint64_t cycles_per_sec){
  /* lambda specifies the number of requests per second */
  std::random_device rd;
  std::mt19937 gen(rd());
  std::exponential_distribution<> d(lambda);

  uint64_t cumulative_cycles = 0;
  for (int i = 0; i < num_reqs; ++i) {
      double interarrival_time = d(gen);
      cumulative_cycles += static_cast<uint64_t>(interarrival_time * cycles_per_sec);
      offsets[i] = cumulative_cycles;
  }

}
#endif

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
  idxd_comp *comp = args->comp;
  int id = args->id;
  unsigned long d_out_spc = IAA_DECOMPRESS_MAX_DEST_SIZE;
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
  while(enqcmd((void *)((char *)(m_iaa->wq_reg) + p_off), desc) ){
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
  if(comp->status != IAX_COMP_SUCCESS && comp->status != IAX_COMP_NONE){
    LOG_PRINT(LOG_ERR, "Decompression for request:%d failed: %d\n", id, comp->status);
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

#ifdef BLOCK
static __always_inline void ddh_block(job_info_t *jinfo, coro_t::push_type & yield){
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
  while(enqcmd((void *)((char *)(m_iaa->wq_reg) + p_off), desc) ){
    /* retry submit */
  }

#ifdef EXETIME
  ts2 = rdtsc();
#endif
  jinfo->s = OFFLOADED;
  while(comp->status == IAX_COMP_NONE){
  }

  if(comp->status != IAX_COMP_SUCCESS && comp->status != IAX_COMP_NONE){
    LOG_PRINT(LOG_ERR, "Decompression for request:%d failed: %d\n", id, comp->status);
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
#endif

#ifdef NOACC
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
#endif


/*
One option is to create a coroutine that yields after each request and
returns the reason it yielded
The coroutine takes its job info and the context to yield to and its ID
*/
void coro(int id, job_info_t *&jinfo, coro_t::push_type &yield)
{
  LOG_PRINT(LOG_DEBUG, "[coro]: coro %d is ready!\n", id);
  yield(&yield);
  for(;;){
    switch(jinfo->jtype){
      case DESER_DECOMP_HASH:
      #ifdef NOACC
        ddh_noacc(jinfo, yield);
      #elif defined(BLOCK)
        ddh_block(jinfo, yield);
      #else
        ddh_yielding(jinfo, yield);
      #endif
        break;
      case DECRYPT_MEMCPY_DP:
        std::cout << "Decrypting, memcpying, and DP" << std::endl;
        break;
      default:
        std::cout << "Unknown job type" << std::endl;
        break;
    }
    jinfo->s = COMPLETED;
    yield(&yield);
  }
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

  /* TODO: create_workload_payloads(remaining_bytes, workload_type) */
  uint64_t bytes = get_free_hugepages() * 1024 * 1024 * 1024;
  uint64_t comp_space_bytes = sizeof(idxd_comp) * num_reqs;
  uint64_t desc_space_bytes = sizeof(idxd_desc) * num_reqs;
  uint64_t args_space_bytes = sizeof(ddh_args_t) * num_reqs;
  uint64_t hdr_space_bytes = sizeof(hdr) * num_reqs;
  uint64_t necessary_bytes = args_space_bytes + comp_space_bytes + desc_space_bytes + hdr_space_bytes;
  if(necessary_bytes > bytes){
    LOG_PRINT(LOG_ERR, "Not enough memory to allocate all the necessary buffers\n");
    return NULL;
  }
  uint64_t remaining_bytes = bytes - necessary_bytes;

  /* per-request items */
  struct numa_mem *nm = (struct numa_mem *)calloc(1, sizeof(struct numa_mem));
  idxd_desc *desc = (idxd_desc *)alloc_numa_offset(nm, num_reqs * sizeof(idxd_desc), 0);
  idxd_comp *comp = (idxd_comp *)alloc_numa_offset(nm, num_reqs * sizeof(idxd_comp), 0);
  hdr *hdrs = (hdr *)alloc_numa_offset(nm, num_reqs * sizeof(hdr), 0);

  /* reusable payload items */
  std::random_device rd;
  std::mt19937 gen(rd());
  std::exponential_distribution<> d(1.0 / 1024);

  std::vector<uint64_t> ser_buf_offsets;
  std::vector<uint64_t> payload_sizes;
  std::vector<uint64_t> max_comp_sizes;
  std::vector<uint64_t> max_payload_sizes;
  uint64_t used_payload_bytes = 0, total_ser_buf_space = 0, total_dbuf_bytes = 0;
  int s_sz;
  int num_bufs = 0;
  double target_ratio = 3.0;
  uint64_t decomp_out_space = IAA_DECOMPRESS_MAX_DEST_SIZE;
  for(;;){
    uint64_t exp_dist_value = static_cast<uint64_t>(d(gen));
    uint64_t decomp_size = exp_dist_value * target_ratio;
    uint64_t max_comp_size = get_compress_bound(decomp_size);
    uint64_t max_expand_bytes = max_comp_size + MAX_SER_OVERHEAD_BYTES;
    max_expand_bytes = (max_expand_bytes + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
    used_payload_bytes += (max_expand_bytes + decomp_out_space);
    if (used_payload_bytes > remaining_bytes || num_bufs == num_reqs){
      used_payload_bytes -= (max_expand_bytes + decomp_out_space);
      break;
    } else {
      max_comp_sizes.push_back(max_comp_size);
      max_payload_sizes.push_back(max_expand_bytes);
      ser_buf_offsets.push_back(total_ser_buf_space);
      payload_sizes.push_back(exp_dist_value);
      total_ser_buf_space += max_expand_bytes;
      total_dbuf_bytes += decomp_out_space;
      LOG_PRINT(LOG_DEBUG, "EXP: %lu SER: %lu TotalSer: %lu TotalDbuf: %lu\n",
        exp_dist_value, max_expand_bytes, total_ser_buf_space, total_dbuf_bytes);
      num_bufs++;
    }
  }
  LOG_PRINT(LOG_DEBUG, "NumBufs: %d\n", num_bufs);
  char *s_bufs = (char *)alloc_numa_offset(nm, total_ser_buf_space, 0);
  char *d_bufs = (char *)alloc_numa_offset(nm, total_dbuf_bytes, 0);
  ddh_args_t *ddh_args = (ddh_args_t *)alloc_numa_offset(nm, num_reqs * sizeof(ddh_args_t), 0);

  int rc = alloc_numa_mem(nm, PAGE_SIZE, server_node);
  add_base_addr(nm, (void **)&desc);
  add_base_addr(nm, (void **)&comp);
  add_base_addr(nm, (void **)&s_bufs);
  add_base_addr(nm, (void **)&d_bufs);
  add_base_addr(nm, (void **)&ddh_args);
  add_base_addr(nm, (void **)&hdrs);

  std::vector<uint64_t> ser_sizes;
  for(int i=0; i<num_reqs; i++){
    uint64_t ser_buf_offset = ser_buf_offsets[(i % num_bufs)];
    uint64_t max_payload_expansion = max_payload_sizes[(i % num_bufs)];
    uint64_t payload_size = payload_sizes[(i % num_bufs)];
    uint64_t max_comp_size = max_comp_sizes[(i % num_bufs)];
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
    ddh_args[i].d_buf = &d_bufs[(i % num_bufs) * decomp_out_space];
    ddh_args[i].d_sz = payload_size * target_ratio;
    ddh_args[i].id = i;
    ddh_args[i].p_off = 0;
  }

  double mean = std::accumulate(payload_sizes.begin(), payload_sizes.end(), 0.0) / payload_sizes.size();
  PRINT("[Workload Gen] Exponential Workload Distribution DDH Attempted Payload Size: %f\n", mean);
  double actual_mean = std::accumulate(ser_sizes.begin(), ser_sizes.end(), 0.0) / ser_sizes.size();
  PRINT("[Workload Gen] Actual Payload Size: %f\n", actual_mean);

  #ifdef POISSON
  uint64_t *offsets = (uint64_t *)malloc(num_reqs * sizeof(uint64_t));
  uint64_t freqHz = 2100000000;
  double max_rps = 1000000;
  double load = args->load;
  PRINT("[Workload Gen] Load: %f MRPS: %f\n", load, load * max_rps);
  gen_interarrival_poisson(max_rps * load, offsets, num_reqs, freqHz);
  for(int i=0; i<num_reqs; i++){
    void *arg = (void *)&ddh_args[i];
    hdrs[i].arrival = offsets[i];
    hdrs[i].id = i;
    hdrs[i].req_type = DESER_DECOMP_HASH;
    hdrs[i].w_args = arg;
    hdrs[i].comp = &comp[i];
  }
  workload_start_cycle = 0;
  #endif

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
    #ifndef LATENCY
    void *arg = (void *)&ddh_args[pushed];
    hdr h {pushed, DESER_DECOMP_HASH, arg};
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
  uint64_t start, end;
  start = fenced_rdtscp();
  #endif
  while(pushed < num_reqs){

    #ifdef LATENCY
    if( pushed % sampling_interval){
      hdrs[pushed].tagged = true;
      tagged_jobs++;
    } else {
      hdrs[pushed].tagged = false;
    }
    #else
    hdr h {(uint32_t)pushed, DESER_DECOMP_HASH, arg};
    #endif
    #ifdef POISSON
    now = rdtsc();
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

  #ifdef BOTTLENECK
  PRINT("[Workload Gen] Failed enqueues: %ld\n", failed_enq);
  #endif

  #ifdef LATENCY
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
  int num_requests = args->num_requests;
  int outq = 0;
  pthread_barrier_t *start_barrier = args->start_barrier;
  hdr h;
  int num_coros = args->num_coros;

  std::vector<coro_info_t*> idle_coros;
  idle_coros.reserve(num_coros);
  coro_t::pull_type *coros
    = (coro_t::pull_type *)malloc(num_coros * sizeof(coro_t::pull_type));
  coro_info_t *c_infos
    = (coro_info_t *)malloc(num_coros * sizeof(coro_info_t));
  job_info_t *job_infos = (job_info_t *)malloc(num_coros * sizeof(job_info_t));

  char *st_pool = args->stack_pool;
  for(int i = 0; i < num_coros; i++){
    coros[i] =
      coro_t::pull_type(
          SimpleStack(static_cast<char*>(st_pool + (i * STACK_SIZE))),
          boost::bind( coro, i, &job_infos[i], _1));

    c_infos[i].coro = &coros[i];
    c_infos[i].jinfo = &job_infos[i];
    c_infos[i].jinfo->s = INIT;
    idle_coros.push_back(&c_infos[i]);
  }

  int **pp_completed  = args->pp_completed;

  uint64_t tagged_jobs=0;
  std::queue<hdr> task_q;
  std::list<coro_info_t *> monitor_q;
  std::list<coro_info_t *> post_q;

  worker_info_t *tmp_w;
  std::priority_queue<worker_info_t*, std::vector<worker_info_t*>,
    decltype(&worker_info_ptr_cmp)>
      worker_queue(worker_info_ptr_cmp);

  for(int i=0; i<num_workers; i++){
    worker_info_t *w_info = (worker_info_t *)calloc(1, sizeof(worker_info_t));
    w_info->dq_idx = i;
    w_info->num_running_jobs = 0;
    w_info->dispatched_jobs = 0;
    w_info->p_finished_jobs = pp_completed[i];
    worker_queue.push(w_info);
  }

  pthread_barrier_wait(start_barrier);



  /* Dequeue from the workload ring and print the requests */
  for(;;){
    /* dequeue */
    while(q->try_dequeue(h)){
      task_q.push(h);
    }


    /* check for completed offloads */
    for(auto it = monitor_q.begin(); it != monitor_q.end();){
      coro_info_t *c_info = *it;
      if(c_info->jinfo->s == COMPLETED){
        idle_coros.push_back(c_info);
        it = monitor_q.erase(it);
      } else if (c_info->comp->status != IAX_COMP_NONE && c_info->jinfo->s == OFFLOADED){
        hdr h;
        h.c_info = c_info;
        h.id = c_info->id;
        h.tagged = c_info->tagged;
        h.arrival = c_info->arrival;
        h.dispatched1 = c_info->dispatched1;
        h.first_served = c_info->first_served;
        h.dispatched2 = c_info->dispatched2;
        h.served_again = c_info->served_again;
        h.completed = c_info->completed;

        tmp_w  = worker_queue.top();
          worker_queue.pop();
          outq = tmp_w->dq_idx;
          tmp_w->num_running_jobs
             = (tmp_w->dispatched_jobs++)
              - *(tmp_w->p_finished_jobs) + 1;
          worker_queue.push(tmp_w);
        _mm_sfence();
        while(!d[outq]->try_enqueue(h)){
          outq = (outq + 1) % num_workers;
        }
        post_q.push_back(c_info);
        it = monitor_q.erase(it);
      } else {
        it++;
      }
    }

    for(;;){
      if(!task_q.empty()){
        /* load balance */
        if(!idle_coros.empty()){
          h = task_q.front();
          task_q.pop();
          coro_info_t *c_info = idle_coros.back();
          idle_coros.pop_back();
          c_info->jinfo->args = h.w_args;
          c_info->jinfo->jtype = h.req_type;
          c_info->comp = h.comp;
          c_info->jinfo->s = INIT;
          c_info->id = h.id;
          c_info->tagged = h.tagged;
          c_info->arrival = h.arrival;
          c_info->dispatched1 = rdtsc();
          c_info->first_served = rdtsc();
          c_info->dispatched2 = 0;
          c_info->served_again = 0;
          c_info->completed = 0;

          h.c_info = c_info;

          tmp_w  = worker_queue.top();
          worker_queue.pop();
          outq = tmp_w->dq_idx;
          tmp_w->num_running_jobs
             = (tmp_w->dispatched_jobs++)
              - *(tmp_w->p_finished_jobs) + 1;
          worker_queue.push(tmp_w);
          _mm_sfence();
          while(!d[outq]->try_enqueue(h)){
            outq = (outq + 1) % num_workers;
          }
          monitor_q.push_back(c_info);

        } else {
          break;
        }
      } else {
        break;
      }
    }

    /* check for completed */
    for(auto it = post_q.begin(); it != post_q.end();){
      coro_info_t *c_info = *it;
      if(c_info->jinfo->s == COMPLETED){
        idle_coros.push_back(c_info);
        it = post_q.erase(it);
      } else {
        it++;
      }
    }
  }

  return NULL;
}

void *worker(void *arg){
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
  pthread_barrier_t *start_barrier = args->start_barrier;
  hdr h;
  int *p_completed = args->p_completed;

  LOG_PRINT(LOG_DEBUG, "Worker %d allocating IAA %d\n", id, iaa_dev_id);
  thread_wq_alloc(&m_iaa, iaa_dev_id);

  pthread_barrier_wait(start_barrier);
  // /* scheduler routine */
  for(;;){
    if(q->try_dequeue(h)){
      LOG_PRINT(LOG_DEBUG, "Worker %d dequeued request %d\n", id, h.id);
      /* get a coroutine */
      coro_info_t *c_info = h.c_info;
      (*c_info->coro)();
      (*p_completed)++;
      if(c_info->jinfo->s == COMPLETED){
        c_q->enqueue(h);
      }

    }
  }

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

  pthread_barrier_wait(args->start_barrier);
  #ifdef THROUGHPUT
  while(num_pulled < warmup){
    dq_idx = (dq_idx + 1) % num_workers;
    if(c_queues[dq_idx]->try_dequeue(h)){
      #ifdef LATENCY
      if(h.tagged){
        uint64_t compl_time = fenced_rdtscp();
        h.completed = compl_time;
        hs[num_pulled ] = h;
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
        hs[num_pulled ] = h;
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
  std::vector<uint64_t> latencies;
  std::vector<uint64_t> t_in_wq2;
  std::vector<uint64_t> t_in_wq;
  std::vector<uint64_t> t_in_dispatchq2;
  std::vector<uint64_t> t_in_dispatchq;
  std::vector<uint64_t> second_service;
  double median;
  for(int i=0; i<tagged_jobs; i++){
    if(hs[i].id != 0){ /* TODO: currently some requests are making it here with unmarked headers -- just discard these for now */
    t_in_dispatchq.push_back(hs[i].dispatched1 - hs[i].arrival);
    t_in_wq.push_back(hs[i].first_served - hs[i].dispatched1);
    t_in_dispatchq2.push_back(hs[i].dispatched2 - hs[i].first_served);
    t_in_wq2.push_back(hs[i].served_again - hs[i].dispatched2);
    second_service.push_back(hs[i].completed - hs[i].served_again);

    uint64_t arrival_cycle = arrival_cycle_0 + hs[i].arrival;
    t_in_dispatchq.push_back(hs[i].dispatched1 - arrival_cycle);
    latencies.push_back(hs[i].completed - arrival_cycle);

      LOG_PRINT(LOG_DEBUG,"[Monitor %d] Id:%ld, "
        "Arrival: %ld, "
        "Dispatched1: %ld, "
        "First Served: %ld, "
        "Dispatched2: %ld, "
        "Served Again: %ld, "
        "Completed: %ld, "
        "Latency: %ld\n", args->id, hs[i].id,
        hs[i].arrival,
        hs[i].dispatched1,
        hs[i].first_served, hs[i].dispatched2, hs[i].served_again,
        hs[i].completed, hs[i].completed - hs[i].arrival);
    }
    // PRINT("[Monitor %d] Injected: %ld, Completed: %ld Latency: %ld\n", args->id, hs[i].injected, hs[i].completed, hs[i].completed - hs[i].injected);
  }

  std::sort(latencies.begin(), latencies.end());
  median = latencies[latencies.size() / 2];
  PRINT("[Monitor %d] Median Latency: %f\n", args->id, median);

  size_t idx_99 = static_cast<size_t>(0.99 * latencies.size());
  double p99 = latencies[idx_99];
  PRINT("[Monitor %d] 99th Percentile Latency: %f\n", args->id, p99);

  double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
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

  std::sort(t_in_wq2.begin(), t_in_wq2.end());
  std::sort(t_in_wq.begin(), t_in_wq.end());
  std::sort(t_in_dispatchq2.begin(), t_in_dispatchq2.end());

  double avg_time_in_wq2 = calculate_average(t_in_wq2);
  double median_time_in_wq2 = calculate_percentile(t_in_wq2, 0.5);
  double p99_time_in_wq2 = calculate_percentile(t_in_wq2, 0.99);

  double avg_time_in_wq = calculate_average(t_in_wq);
  double median_time_in_wq = calculate_percentile(t_in_wq, 0.5);
  double p99_time_in_wq = calculate_percentile(t_in_wq, 0.99);

  double avg_time_in_dispatchq2 = calculate_average(t_in_dispatchq2);
  double median_time_in_dispatchq2 = calculate_percentile(t_in_dispatchq2, 0.5);
  double p99_time_in_dispatchq2 = calculate_percentile(t_in_dispatchq2, 0.99);

  PRINT("[Monitor %d] Average Time in Worker Queue 2: %f\n", args->id, avg_time_in_wq2);
  PRINT("[Monitor %d] Median Time in Worker Queue 2: %f\n", args->id, median_time_in_wq2);
  PRINT("[Monitor %d] 99th Percentile Time in Worker Queue 2: %f\n", args->id, p99_time_in_wq2);


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

  PRINT("[Monitor %d] Average Time in Dispatch Queue 2: %f\n", args->id, avg_time_in_dispatchq2);
  PRINT("[Monitor %d] Median Time in Dispatch Queue 2: %f\n", args->id, median_time_in_dispatchq2);
  PRINT("[Monitor %d] 99th Percentile Time in Dispatch Queue 2: %f\n", args->id, p99_time_in_dispatchq2);
  #endif

  for (int i = 0; i < num_workers; i++) {
    PRINT("[Monitor %d] Worker %d processed %lu requests\n", args->id, i, cs_per_wrkr[i]);
  }
  free(cs_per_wrkr);

  PRINT("Config: ");
  PRINT("requests: %d ", expected);
  PRINT("workers: %d ", num_workers);
  PRINT("coros_per_worker: %d ", args->coros_per_worker);
  PRINT("IAAs: %d ", args->num_iaas);
  PRINT("Inqueue_size: %d ", args->inqueue_size_elems);
  PRINT("Dispatch_queue_size: %d\n", args->dispatch_queue_num_elems);

  exit(0);
  pthread_barrier_wait(args->exit_barrier);
  // we know everything is done -- exit
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
  // int *occupancy_lines = (int *)malloc(num_workers * CACHE_LINE_SIZE);

  pthread_barrier_t *exit_barrier = (pthread_barrier_t *)malloc(sizeof(pthread_barrier_t));
  pthread_barrier_t *start_barrier = (pthread_barrier_t *)malloc(sizeof(pthread_barrier_t));

  int opt;
  while((opt = getopt(argc, argv, "n:c:q:d:i:w:l:s:y:u:p:")) != -1){
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
  int **pp_completed = (int **)malloc(num_workers * sizeof(int *));
  for(int i=0; i<num_workers; i++){
    pp_completed[i] = (int *)aligned_alloc(CACHE_LINE_SIZE, CACHE_LINE_SIZE);
    pp_completed[i][0] = 0;
  }
  #ifdef JSQ
  int **pp_finished_jobs = (int **)malloc(num_workers * sizeof(int *));
  for(int i=0; i<num_workers; i++){
    pp_finished_jobs[i] = (int *)aligned_alloc(CACHE_LINE_SIZE, CACHE_LINE_SIZE);
    *(pp_finished_jobs[i]) = 0;
  }
  #endif

  struct numa_mem *stack_nm = (struct numa_mem *)calloc(1, sizeof(struct numa_mem));
  reserve_numa_mem(stack_nm, num_workers * coros_per_worker * STACK_SIZE, server_node);
  int rc = alloc_numa_mem(stack_nm, PAGE_SIZE, server_node);
  if(rc != 0){
    LOG_PRINT(LOG_ERR, "Error allocating stack memory\n");
    return -1;
  }



  pthread_barrier_init(exit_barrier, NULL, 3);
  pthread_barrier_init(start_barrier, NULL, num_workers + 3);

  dispatch_args->wqs = dispatch_q;
  dispatch_args->inq = wrkload_q;
  dispatch_args->num_workers = num_workers;
  dispatch_args->num_requests = num_reqs;
  dispatch_args->exit_barrier = exit_barrier;
  dispatch_args->num_requests = num_reqs;
  dispatch_args->start_barrier = start_barrier;
  dispatch_args->num_coros = coros_per_worker * num_workers;
  dispatch_args->stack_pool = (char *)stack_nm->base_addr;
  dispatch_args->pp_completed = pp_completed;
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
  #ifdef POISSON
  wrkld_args->load = load;
  #endif
  dispatch_queue_num_elems = dispatch_queue_size_bytes / sizeof(hdr);
  completion_queue_num_elems = completion_queue_size_bytes / sizeof(hdr);
  for(int i=0; i<num_workers; i++){
    dispatch_q[i] = new dispatch_ring_t(dispatch_queue_num_elems);
    c_queues[i] = new response_ring_t(completion_queue_num_elems);
  }

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
    wrk_args[i]->node = server_node;
    wrk_args[i]->start_barrier = start_barrier;
    wrk_args[i]->exit_barrier = exit_barrier;
    wrk_args[i]->stack_pool = (char *)stack_nm->base_addr + (i * coros_per_worker * STACK_SIZE);
    wrk_args[i]->p_completed = pp_completed[i];
    #ifdef JSQ
    wrk_args[i]->p_finished_jobs = pp_finished_jobs[i];
    #endif
    create_thread_pinned(&worker_threads[i], worker, static_cast<void*>(wrk_args[i]), worker_start_core + i);
  }

    monitor_args = (monitor_args_t *)malloc(sizeof(monitor_args_t));
    monitor_args->start_barrier = start_barrier;
    monitor_args->exit_barrier = exit_barrier;
    monitor_args->c_queues = c_queues;
    monitor_args->num_requests = num_reqs;
    monitor_args->num_workers = num_workers;
    monitor_args->coros_per_worker = coros_per_worker;
    monitor_args->num_iaas = num_iaas;
    monitor_args->inqueue_size_elems = inqueue_size_elems;
    monitor_args->dispatch_queue_num_elems = dispatch_queue_num_elems;


    monitor_args->id = num_workers;
    create_thread_pinned(&monitor_thread, monitor, static_cast<void*>(monitor_args), monitor_core);


  create_thread_pinned(&wrlkd_thread, workload_gen, static_cast<void *>(wrkld_args), wrkload_gen_core);
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