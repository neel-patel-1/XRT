#include "numa_mem.h"
#include "payload_gen.h"
#include "print_utils.h"
#include "stats.h"
#include "pointer_chase.h"
#include "iaa_offloads.h"
#include "dsa_offloads.h"
#include "sequential_writer.h"
#include <x86intrin.h>
#include <immintrin.h>
#include <sched.h>
#include <sys/sysinfo.h>
#include "proto_files/router.pb.h"
#include "ch3_hash.h"
#include "gather.h"
#include "payload_gen.h"
#include "gpcore_compress.h"
#include "lzdatagen.h"
#include "decrypt.h"
#include "dotproduct.h"
#include "test.h"
#include "gather.h"
#include "pointer_chase.h"
#include "iaa_offloads.h"
#include "dsa_offloads.h"
#include <x86intrin.h>
#include <immintrin.h>
#include "racer_opts.h"
#include "racer_thread.h"
extern "C" {
  #include "fcontext.h"
  #include "idxd.h"
  #include "accel_test.h"
}

typedef struct hw_desc idxd_desc;
typedef struct completion_record idxd_comp;

extern struct acctest_context *iaa;

_Atomic bool do_demote = false;
__thread volatile uint8_t *preempt_signal;
IppsAES_GCMState *pState = NULL;

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

bool gDebugParam = false;
int *glob_indir_arr = NULL; // TODO
int num_accesses = 0; // TODO

int main(int argc, char **argv){

pthread_t emul_thread;
  emul_thread_args_t emul_thread_args;
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
  char *emul_dymmy_buf = NULL;
  char *emul_dymmy_buf_2 = NULL;

  char *tmp_plain_buf = NULL;
  uint64_t src_buf_space;

  get_opts(argc, argv);

  double target_ratio = 3.0;
  uint64_t decomp_size = buf_size * target_ratio;
  uint64_t decomp_out_space = IAA_DECOMPRESS_MAX_DEST_SIZE;
  uint64_t max_comp_size = get_compress_bound(decomp_size);
  uint64_t max_payload_expansion = max_comp_size + 64;

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
  srcs = (char *)alloc_numa_offset(nm, max_payload_expansion * total_requests, 0);
  dsts = (char *)alloc_numa_offset(nm, total_requests * IAA_DECOMPRESS_MAX_DEST_SIZE, 0);
  dsa_dsts = (char *)alloc_numa_offset(nm, total_requests * IAA_DECOMPRESS_MAX_DEST_SIZE, 0);
  emul_dymmy_buf = (char *)alloc_numa_offset(nm, buf_size * total_requests, 0);
  emul_dymmy_buf_2 = (char *)alloc_numa_offset(nm, buf_size * total_requests, 0);


  rc = alloc_numa_mem(nm, pg_size, node);
  if(rc){
    LOG_PRINT(LOG_ERR,"Failed to allocate memory\n");
    return -1;
  }
  add_base_addr(nm, (void **)&srcs);
  add_base_addr(nm, (void **)&dsts);
  add_base_addr(nm, (void **)&dsa_dsts);
  add_base_addr(nm, (void **)&desc);
  add_base_addr(nm, (void **)&comp);
  add_base_addr(nm, (void **)&emul_dymmy_buf);
  add_base_addr(nm, (void **)&emul_dymmy_buf_2);
  memset(comp, 0, sizeof(idxd_comp) * total_requests);
  memset(desc, 0, sizeof(idxd_desc) * total_requests);
  initialize_iaa_wq(iaa_dev_id, iaa_wq_id, wq_type);

  /* pin to SMT threads on same core */
  cpu_pin(3);

  uint64_t buf_offset = (uintptr_t) dsa_dsts - (uintptr_t) dsts;
  init_emul_thread(buf_offset, (uintptr_t)emul_dymmy_buf, (uintptr_t)emul_dymmy_buf_2);

  tmp_plain_buf = (char *)aligned_alloc(64, decomp_size);

#ifdef THROUGHPUT
for(int j=0; j<iter; j++){
  start = rdtsc();
#endif

  flush_range((void *)dsts, IAA_DECOMPRESS_MAX_DEST_SIZE * total_requests); /* flush decrypt dsts */
  for(int i=0; i<IAA_DECOMPRESS_MAX_DEST_SIZE * total_requests; i+=4096){
    dsa_dsts[i] = 'a';
  } /* flush source, write prefault dsat dsts */
  flush_range((void *)emul_dymmy_buf, buf_size * total_requests);

  for(int i=0; i<total_requests; i++){
    void **src = (void **)&srcs[i * max_payload_expansion];
    void **dst = (void **)&dsts[i * IAA_DECOMPRESS_MAX_DEST_SIZE];
    idxd_comp *m_comp = &comp[i];
    idxd_desc *m_desc = &desc[i];
    p_off = (p_off + 64) % 4096;

    int avail_comp_space = max_payload_expansion;
    uLong avail_decomp_space = IAA_DECOMPRESS_MAX_DEST_SIZE;
    /* set up the indices for the pointer chain */
    void **pointer_chain = (void **)tmp_plain_buf;
    uint64_t len = decomp_size / 64;
    uint64_t  *indices = (uint64_t *)malloc(sizeof(uint64_t) * len);
    for (int i = 0; i < len; i++) {
      indices[i] = i;
    }
    random_permutation(indices, len);

    /* put some 3.0 ratio data in the tmp uncomp buf*/
    lzdg_generate_data((void *)tmp_plain_buf, decomp_size, 3.0, 3.0, 3.0);
    /*actually write the pointer chain */
    for(int i=1; i<len; ++i){
      /* pchain at the last index equals the address of the st of the cacheline at the next index*/
      pointer_chain[indices[i-1] * 8] = (void *)&dst[indices[i] * 8];
    }
    /* close the chain loop */
    pointer_chain[indices[len - 1] * 8] = (void *)&dst[indices[0] * 8];

    rc = gpcore_do_compress((void *)src, (void *) tmp_plain_buf, decomp_size, &avail_comp_space);
    if(rc){
      LOG_PRINT(LOG_ERR, "Failed to compress: %d\n", rc);
      return -1;
    }
    LOG_PRINT(LOG_DEBUG, "Compressed size: %d\n", avail_comp_space);

    flush_range((void *)src, max_payload_expansion);

    if(!noAcc){
      /* offload memcpy to dsa */
      prepare_iaa_decompress_desc_with_preallocated_comp(
        m_desc, (uint64_t)(src), (uint64_t)dst,
        (uint64_t)(m_comp), avail_comp_space);
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
      if(m_comp->iax_output_size != (unsigned)decomp_size){
        LOG_PRINT(LOG_ERR, "Failed to decompress: %d\n", m_comp->iax_output_size);
        return -1;
      }
    } else {
      memcpy((void *)dst, (void *)pointer_chain, buf_size);
      rc = gpcore_do_decompress(
        (void *)dst,
        (void *)src,
        avail_comp_space,
        &avail_decomp_space
      );
      #ifdef EXETIME
      end = rdtsc();
      #endif
      if(rc){
        LOG_PRINT(LOG_ERR, "Failed to decompress: %d\n", rc);
        return -1;
      }
      if(avail_decomp_space != decomp_size){
        LOG_PRINT(LOG_ERR, "Failed to decompress: %lu\n", avail_decomp_space);
        return -1;
      }
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
      for(int j=0; j<decomp_size; j+=64){
        _mm_prefetch((void *)(fetch_buf + j), _MM_HINT_T1);
      }
      #ifdef EXETIME
      end = rdtsc();
      pref_array_start[i] = start;
      pref_array_end[i] = end;
    #endif
    }



    float score;
    int sz;
    #ifdef EXETIME
    LOG_PRINT(LOG_TOO_VERBOSE, "Please fetch: %lx\n", (uintptr_t)dst);
    start = rdtsc();
    #endif
    chase_pointers((void **)dst, decomp_size/64);
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

  if(conc_demote ){
    run = false;
    pthread_join(emul_thread, NULL);
  }

  return 0;
}