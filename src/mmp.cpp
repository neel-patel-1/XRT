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
#include <cmath>
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

extern std::atomic<uintptr_t> pref_buf_st_addr;
extern std::atomic<bool> run;
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

  uint64_t start, end;
  int dsa_wq_id = 0;
  int wq_type = SHARED;
  int dsa_dev_id = 0;

  struct numa_mem  *nm = NULL;
  int nb_numa_node = 1;
  int node = 0;

  int rc;
  idxd_desc *desc = NULL;
  idxd_comp *comp = NULL;
  int pg_size = 1024 * 1024 * 1024;
  int p_off = 0;
  char *srcs = NULL;
  char *srcs2 = NULL;
  char *dsts = NULL;
  char *dsa_dsts = NULL;
  int *mean_vector = NULL;
  int batch_size = 64;
  char *emul_dymmy_buf = NULL;
  char *emul_dymmy_buf_2 = NULL;
  mm_data_t *mat_data = NULL;


  get_opts(argc, argv);

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

  memset(demote_array_end, 0, sizeof(uint64_t) * total_requests);
  memset(demote_array_start, 0, sizeof(uint64_t) * total_requests);
  memset(pref_array_end, 0, sizeof(uint64_t) * total_requests);
  memset(pref_array_start, 0, sizeof(uint64_t) * total_requests);

  uint64_t diffs[total_requests];
  uint64_t start_times[iter];
  uint64_t end_times[iter];

  uint64_t bytes = batch_size;

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
  status = ippsAES_GCMInit(pKey, keysize, pState, ippAES_GCM_ctx_size);
  if(status != ippStsNoErr){
    LOG_PRINT(LOG_ERR, "Failed to init AES GCM\n");
  }

  struct completion_record *dummy = (struct completion_record *)malloc(sizeof(struct completion_record));
  dummy->status = IAX_COMP_NONE;
  preempt_signal = (uint8_t *)dummy;

  int matrix_len = floor(sqrt(buf_size/sizeof(int)));
  int mat_size_bytes = matrix_len * matrix_len * sizeof(int);

  nm = (struct numa_mem *)calloc(nb_numa_node, sizeof(struct numa_mem));
  desc = (idxd_desc *)alloc_numa_offset(nm, sizeof(idxd_desc) * total_requests, 0);
  comp = (idxd_comp *)alloc_numa_offset(nm, sizeof(idxd_comp) * total_requests, 0);
  mat_data = (mm_data_t *)alloc_numa_offset(nm,sizeof(mm_data_t) * total_requests, 0);
  srcs = (char *)alloc_numa_offset(nm, mat_size_bytes * total_requests, 0);
  srcs2 = (char *)alloc_numa_offset(nm, mat_size_bytes * total_requests, 0);
  dsts = (char *)alloc_numa_offset(nm, total_requests * mat_size_bytes, 0);
  dsa_dsts = (char *)alloc_numa_offset(nm, total_requests * mat_size_bytes, 0);
  mean_vector = (int *)alloc_numa_offset(nm, total_requests * matrix_len * sizeof(int), 0);
  emul_dymmy_buf = (char *)alloc_numa_offset(nm, buf_size * total_requests, 0);
  emul_dymmy_buf_2 = (char *)alloc_numa_offset(nm, buf_size * total_requests, 0);

  rc = alloc_numa_mem(nm, pg_size, node);
  if(rc){
    LOG_PRINT(LOG_ERR,"Failed to allocate memory\n");
    return -1;
  }
  add_base_addr(nm, (void **)&mat_data);
  add_base_addr(nm, (void **)&srcs);
  add_base_addr(nm, (void **)&srcs2);
  add_base_addr(nm, (void **)&dsts);
  add_base_addr(nm, (void **)&dsa_dsts);
  add_base_addr(nm, (void **)&mean_vector);
  add_base_addr(nm, (void **)&desc);
  add_base_addr(nm, (void **)&comp);
  add_base_addr(nm, (void **)&emul_dymmy_buf);
  add_base_addr(nm, (void **)&emul_dymmy_buf_2);
  memset(comp, 0, sizeof(idxd_comp) * total_requests);
  memset(desc, 0, sizeof(idxd_desc) * total_requests);
  initialize_dsa_wq(dsa_dev_id, dsa_wq_id, wq_type);

  /* pin to SMT threads on same core */
  cpu_pin(3);


#ifdef THROUGHPUT
for(int j=0; j<iter; j++){
  start = rdtsc();
#endif

  flush_range((void *)srcs, mat_size_bytes * total_requests);
  flush_range((void *)dsts, mat_size_bytes * total_requests); /* flush decrypt dsts */
  for(int i=0; i<mat_size_bytes * total_requests; i+=4096){
    dsa_dsts[i] = 'a';
  } /* flush source, write prefault dsat dsts */
  flush_range((void *)emul_dymmy_buf, buf_size * total_requests);

  for(int i=0; i<total_requests; i++){
    int *src = (int *)&srcs[i * mat_size_bytes];
    int *src2 = (int *)&srcs2[i * mat_size_bytes];
    int *dst = (int *)&dsts[i * mat_size_bytes];
    int *mean_vec = (int *)&mean_vector[i * matrix_len];
    idxd_comp *m_comp = &comp[i];
    idxd_desc *m_desc = &desc[i];
    p_off = (p_off + 64) % 4096;

    for(int k=0; k<matrix_len; k++){
      for(int l=0; l<matrix_len; l++){
        if(k == l){
          src[k * matrix_len + l] = 1;
          src2[k * matrix_len + l] = 1;
        } else {
          src[k * matrix_len + l] = 0;
          src2[k * matrix_len + l] = 0;
        }
      }
    }

    mm_data_t *mm_data = &mat_data[i];
    mm_data->matrix_A = (int *)src;
    mm_data->matrix_B = (int *)src2;
    mm_data->matrix_out = (int *)dst;
    mm_data->matrix_len = matrix_len;

    // write_to_buf((char *)dst, buf_size);
      #ifdef EXETIME
      start = rdtsc();
      #endif

      matrix_mult(mm_data);

      #ifdef EXETIME
      end = rdtsc();
      #endif

    #ifdef EXETIME
    preproc_array_start[i] = start;
    preproc_array_end[i] = end;
    #endif

    if(sync_demote){
      #ifdef EXETIME
      start = rdtsc();
      #endif
      demote_buf((char *)(mm_data->matrix_out), mat_size_bytes);
      #ifdef EXETIME
      end = rdtsc();
      demote_array_start[i] = start;
      demote_array_end[i] = end;
      #endif
    }
    /* offload memcpy to dsa */
    prepare_dsa_memfill_desc_with_preallocated_comp(
        m_desc,
        0,
        (uint64_t )(mm_data->matrix_out),
        (uint64_t)m_comp,
        mat_size_bytes/2
    );
    while(enqcmd((void *)((char *)(dsa->wq_reg) + p_off), m_desc) ){
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

    #ifdef EXETIME
    memcpy_array_start[i] = start;
    memcpy_array_end[i] = end;
    #endif


    if(m_comp->status != IAX_COMP_SUCCESS){
      LOG_PRINT(LOG_ERR,"Failed to copy: %x\n", comp[i].status);
      return -1;
    }

    if(sync_prefetch){
      char *fetch_buf = (char *)(mm_data->matrix_out);
      #ifdef EXETIME
      start = rdtsc();
      #endif
      for(int j=0; j<mat_size_bytes; j+=64){
        _mm_prefetch((void *)(fetch_buf + j), _MM_HINT_T1);
      }
      #ifdef EXETIME
      end = rdtsc();
      pref_array_start[i] = start;
      pref_array_end[i] = end;
    #endif
    }

    #ifdef EXETIME
    LOG_PRINT(LOG_DEBUG, "Please fetch: %lx\n", (uintptr_t)mm_data->matrix_out);
    start = rdtsc();
    #endif
    calc_mean(
      (int *)mm_data->matrix_out,
      (int *)mean_vec,
      matrix_len,
      matrix_len
    );
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
#endif

LOG_PRINT(LOG_DEBUG, "Buffer size: %lu\n", buf_size);
LOG_PRINT(LOG_DEBUG, "Total requests: %lu\n", total_requests);

#ifdef THROUGHPUT
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