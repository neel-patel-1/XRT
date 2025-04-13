#include "numa_mem.h"
#include "payload_gen.h"
#include "print_utils.h"
#include "stats.h"
#include "pointer_chase.h"
#include "iaa_offloads.h"
#include "dsa_offloads.h"
#include <x86intrin.h>
#include <immintrin.h>
#include <sched.h>
#include <sys/sysinfo.h>
#include "src/protobuf/generated/src/protobuf/router.pb.h"
#include "ch3_hash.h"
#include "gather.h"
#include "matmul_histogram.h"
#include "payload_gen.h"
#include "gpcore_compress.h"
#include "lzdatagen.h"
#include "decrypt.h"
#include "dotproduct.h"
#include "test.h"
#include "gather.h"
#include "pointer_chase.h"
#include "timer_utils.h"
#include "iaa_offloads.h"
#include "dsa_offloads.h"
#include <x86intrin.h>
#include <immintrin.h>
#include "racer_opts.h"
#include "gather.h"
#include <atomic>
extern "C" {
  #include "fcontext.h"
  #include "idxd.h"
  #include "accel_test.h"
}

#define COMP_STATUS_COMPLETED 1
#define COMP_STATUS_PENDING 0
#define REQUEST_STATUS_NONE -1
#define REQUEST_COMPLETED COMP_STATUS_COMPLETED
#define REQUEST_PREEMPTED COMP_STATUS_PENDING
#define REQUEST_YIELDED 2
typedef struct _request_return_args{
  int status;
} request_return_args_t;

typedef struct hw_desc idxd_desc;
typedef struct completion_record idxd_comp;

extern struct acctest_context *iaa;

std::atomic<bool> do_demote = false;
__thread volatile uint8_t *preempt_signal;
__thread IppsAES_GCMState *pState = NULL;

__thread request_return_args_t ret_args;
__thread fcontext_transfer_t scheduler_xfer;

extern "C" void do_yield()
{
  LOG_PRINT(LOG_DEBUG, "Yielding\n");
  ret_args.status = REQUEST_PREEMPTED;
  scheduler_xfer = fcontext_swap(scheduler_xfer.prev_context, NULL);
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


static __always_inline void flush_range(void *start, size_t len)
{
  char *ptr = (char *)start;
  char *end = ptr + len;

  for (; ptr < end; ptr += 64) {
    _mm_clflush(ptr);
  }
}

static __always_inline void demote_buf(char *buf, int size){
  for(int i = 0; i < size; i+=64){
    _cldemote((void *)&buf[i]);
  }

}

static __always_inline void write_to_buf(char *buf, int size){
  for(int i = 0; i < size; i+=64){
    ((volatile char *)(buf))[i] = 'a';
  }
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

void indirect_array_populate(int *indirect_array, int num_accesses, int min_val, int max_val){
  // int num_floats_in_array = array_size_bytes / sizeof(float);
  // int max_val = num_floats_in_array - 1;
  // int min_val = 0;
  for(int i=0; i<num_accesses; i++){
    int idx = (rand() % (max_val - min_val + 1)) + min_val;
    indirect_array[i] = idx;
  }
}


bool gDebugParam = false;
int *glob_indir_arr = NULL; // TODO
int num_accesses = 0; // TODO

int main(int argc, char **argv){

  uint64_t start, end;

  int iaa_wq_id = 0;
  int wq_type = SHARED;
  int iaa_dev_id = 1;

  struct numa_mem  *nm = NULL;
  int nb_numa_node = 1;
  int node = 0;

  int rc;
  uint64_t iter = 1;
  idxd_desc *desc = NULL;
  idxd_comp *comp = NULL;
  int pg_size = 1024 * 1024 * 1024;
  int p_off = 0;
  char *srcs = NULL;
  char *dsts = NULL;
  char *dsa_dsts = NULL;
  int batch_size = 64;

  int *scat_bufs = NULL;
  int *hists = NULL;
  int hist_size = 256 * 3;

  uint8_t *aecs = NULL;

  get_opts(argc, argv);

  int max_indirect_index = buf_size / sizeof(float) - 1;
  int num_accesses = buf_size / sizeof(float);

  uint64_t memcpy_array_start[total_requests];
  uint64_t memcpy_array_end[total_requests];
  uint64_t preproc_array_start[total_requests];
  uint64_t demote_array_start[total_requests];
  uint64_t demote_array_end[total_requests];
  uint64_t preproc_array_end[total_requests];
  uint64_t pref_array_start[total_requests];
  uint64_t pref_array_end[total_requests];
  uint64_t dotprod_array_start[total_requests];
  uint64_t dotprod_array_end[total_requests];

  memset(memcpy_array_start, 0, sizeof(uint64_t) * total_requests);
  memset(memcpy_array_end, 0, sizeof(uint64_t) * total_requests);
  memset(preproc_array_start, 0, sizeof(uint64_t) * total_requests);
  memset(preproc_array_end, 0, sizeof(uint64_t) * total_requests);
  memset(demote_array_start, 0, sizeof(uint64_t) * total_requests);
  memset(demote_array_end, 0, sizeof(uint64_t) * total_requests);
  memset(pref_array_start, 0, sizeof(uint64_t) * total_requests);
  memset(pref_array_end, 0, sizeof(uint64_t) * total_requests);
  memset(dotprod_array_start, 0, sizeof(uint64_t) * total_requests);
  memset(dotprod_array_end, 0, sizeof(uint64_t) * total_requests);

  uint64_t diffs[total_requests];
  uint64_t start_times[iter];
  uint64_t end_times[iter];

  uint64_t bytes = batch_size;

  struct completion_record *dummy = (struct completion_record *)malloc(sizeof(struct completion_record));
  dummy->status = IAX_COMP_NONE;
  preempt_signal = (uint8_t *)dummy;

  nm = (struct numa_mem *)calloc(nb_numa_node, sizeof(struct numa_mem));
  desc = (idxd_desc *)alloc_numa_offset(nm, sizeof(idxd_desc) * total_requests, 0);
  comp = (idxd_comp *)alloc_numa_offset(nm, sizeof(idxd_comp) * total_requests, 0);
  srcs = (char *)alloc_numa_offset(nm, buf_size * total_requests, 0);
  dsts = (char *)alloc_numa_offset(nm, total_requests * IAA_DECOMPRESS_MAX_DEST_SIZE, 0);
  hists = (int *)alloc_numa_offset(nm, sizeof(int) * hist_size * total_requests, 0);
  scat_bufs = (int *)alloc_numa_offset(nm, sizeof(int) * num_accesses * total_requests, 0);
  dsa_dsts = (char *)alloc_numa_offset(nm, total_requests * IAA_DECOMPRESS_MAX_DEST_SIZE, 0);
  aecs = (uint8_t *)alloc_numa_offset(nm, IAA_FILTER_AECS_SIZE * 2 * total_requests, 0);

  rc = alloc_numa_mem(nm, pg_size, node);
  if(rc){
    LOG_PRINT(LOG_ERR,"Failed to allocate memory\n");
    return -1;
  }
  add_base_addr(nm, (void **)&srcs);
  add_base_addr(nm, (void **)&dsts);
  add_base_addr(nm, (void **)&hists);
  add_base_addr(nm, (void **)&scat_bufs);
  add_base_addr(nm, (void **)&dsa_dsts);
  add_base_addr(nm, (void **)&desc);
  add_base_addr(nm, (void **)&comp);
  add_base_addr(nm, (void **)&aecs);
  memset(comp, 0, sizeof(idxd_comp) * total_requests);
  memset(desc, 0, sizeof(idxd_desc) * total_requests);
  initialize_iaa_wq(iaa_dev_id, iaa_wq_id, wq_type);

  /* pin to SMT threads on same core */
  cpu_pin(3);


#ifdef THROUGHPUT
for(int j=0; j<iter; j++){
  start = rdtsc();
#endif

  flush_range((void *)dsts, IAA_DECOMPRESS_MAX_DEST_SIZE * total_requests); /* flush decrypt dsts */
  for(int i=0; i<IAA_DECOMPRESS_MAX_DEST_SIZE * total_requests; i+=4096){
    dsa_dsts[i] = 'a';
  } /* flush source, write prefault dsat dsts */

  for(int i=0; i<total_requests; i++){
    indirect_array_populate((int *)&scat_bufs[i * num_accesses], num_accesses, 0, buf_size - 1);
  }

  for(int i=0; i<total_requests; i++){
    uint8_t *src = (uint8_t *)&srcs[i * buf_size];
    uint8_t *dst = (uint8_t *)&dsts[i * IAA_DECOMPRESS_MAX_DEST_SIZE];
    uint8_t *m_aecs = (uint8_t *)&aecs[i * IAA_FILTER_AECS_SIZE * 2];
    int *scat_buf = (int *)&scat_bufs[i * num_accesses];
    int *hist = (int *)&hists[i * hist_size];
    idxd_comp *m_comp = &comp[i];
    idxd_desc *m_desc = &desc[i];
    p_off = (p_off + 64) % 4096;

    #ifdef EXETIME
    start = rdtsc();
    #endif

    scatter_update_inplace_using_indir_array(src, scat_buf, num_accesses);

    #ifdef EXETIME
    end = rdtsc();
    #endif
    #ifdef EXETIME
    preproc_array_start[i] = start;
    preproc_array_end[i] = end;
    #endif

    if(sync_demote){
      LOG_PRINT(LOG_TOO_VERBOSE, "Sync demote real buffers\n");
    #ifdef EXETIME
      start = rdtsc();
      #endif
      demote_buf((char *)&(src), buf_size);
      #ifdef EXETIME
      end = rdtsc();
      demote_array_end[i] = end;
      demote_array_start[i] = start;
      #endif
    } else {
      #ifdef EXETIME
      demote_array_end[i] = 0;
      demote_array_start[i] = 0;
      #endif
    }


    if(!noAcc){
      /* offload memcpy to dsa */
      prepare_iaa_filter_desc_with_preallocated_comp(
        m_desc, (uint64_t)src, (uint64_t)dst, (uint64_t)m_comp, buf_size);
      while(enqcmd((void *)((char *)(iaa->wq_reg) + p_off), m_desc) ){
        /* retry submit */
      }

      #ifdef EXETIME
      start = rdtsc();
      #endif
      while(m_comp->status == 0){

      }
      #ifdef EXETIME
      end = rdtsc();
      #endif
      if(m_comp->status != IAX_COMP_SUCCESS){
        LOG_PRINT(LOG_ERR,"Failed to decomp: %x\n", comp[i].status);
        return -1;
      }
    } else {
      #ifdef EXETIME
      start = rdtsc();
      #endif
      gpcore_do_extract(
        src,
        dst,
        0,
        buf_size / 2,
        m_aecs
      );
      #ifdef EXETIME
      end = rdtsc();
      #endif
    }

    #ifdef EXETIME
    memcpy_array_start[i] = start;
    memcpy_array_end[i] = end;
    #endif

    if(sync_prefetch){
      LOG_PRINT(LOG_TOO_VERBOSE, "Sync prefetch real buffers\n");
      char *fetch_buf = (char *)dst;
      #ifdef EXETIME
      start = rdtsc();
      #endif
      for(int j=0; j<buf_size; j+=64){
        _mm_prefetch((void *)(fetch_buf + j), _MM_HINT_T1);
      }
      #ifdef EXETIME
      end = rdtsc();
      pref_array_start[i] = start;
      pref_array_end[i] = end;
    #endif
    }



    int sz;
    #ifdef EXETIME
    LOG_PRINT(LOG_TOO_VERBOSE, "Please fetch: %lx\n", (uintptr_t)dst);
    start = rdtsc();
    #endif

    calc_hist(dst, hist, buf_size/2, &sz);

    #ifdef EXETIME
    end = rdtsc();
    #endif

    #ifdef EXETIME
    dotprod_array_start[i] = start;
    dotprod_array_end[i] = end;
    #endif
  }
#ifdef THROUGHPUT
  end = rdtsc();
  start_times[j] = start;
  end_times[j] = end;
}
uint64_t exe_time_diffs[iter];
uint64_t exe_times_avg;
avg_samples_from_arrays(exe_time_diffs, exe_times_avg, end_times, start_times, iter);
PRINT("%lu ", buf_size);
mean_median_stdev_rps(exe_time_diffs, iter, total_requests, " RPS");

#endif

  #ifdef EXETIME
  uint64_t PreProcAvg;
  uint64_t MemcpyAvg;
  uint64_t DemoteAvg;
  uint64_t SwPrftchAvg;
  uint64_t DotProdAvg;
  avg_samples_from_arrays(diffs,PreProcAvg, preproc_array_end, preproc_array_start, total_requests);
  avg_samples_from_arrays(diffs,MemcpyAvg, memcpy_array_end, memcpy_array_start, total_requests);
  avg_samples_from_arrays(diffs,DemoteAvg, demote_array_end, demote_array_start, total_requests);
  avg_samples_from_arrays(diffs,SwPrftchAvg, pref_array_end, pref_array_start, total_requests);
  avg_samples_from_arrays(diffs,DotProdAvg, dotprod_array_end, dotprod_array_start, total_requests);

  PRINT("%lu\n%lu\n%lu\n%lu\n%lu\n",PreProcAvg, DemoteAvg, MemcpyAvg, SwPrftchAvg, DotProdAvg );
  #endif


  return 0;
}