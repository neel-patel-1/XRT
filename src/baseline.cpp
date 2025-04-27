#include "print_utils.h"
extern "C" {
  #include "fcontext.h"
  #include "idxd.h"
  #include "accel_test.h"
}

int gLogLevel = LOG_DEBUG;
int *glob_indir_arr = NULL; // TODO
int num_accesses = 0; // TODO
extern struct acctest_context *iaa;
int main(int argc, char **argv){
  fcontext_fn_t m_rq_fn = NULL;
  int arg_len = 0;
  char *m_args = NULL;
  int rc;
  uint64_t total_requests = 10;
  int iter = 10;

}