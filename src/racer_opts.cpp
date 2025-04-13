#include "racer_opts.h"
#include <getopt.h>
#include <stdint.h>
#include <stdlib.h>
bool sync_demote = false;

bool sync_prefetch = false;


uint64_t total_requests = 100;
uint64_t buf_size = 1024;
uint64_t iter = 1;

int gLogLevel = 0;

bool noAcc = false;


int get_opts(int argc, char **argv){
  int opt;
  gLogLevel = LOG_DEBUG;
  while((opt = getopt(argc, argv, "s:nl:t:b:p:i:g")) != -1){
    switch(opt){
      case 's':
        if(atoi(optarg) == 0){
          LOG_PRINT(LOG_DEBUG, "Sync demote\n");
          sync_demote = true;
        } else if(atoi(optarg) == 1){
          LOG_PRINT(LOG_DEBUG, "Sync prefetch\n");
          sync_prefetch = true;
        } else if(atoi(optarg) == 2){
          LOG_PRINT(LOG_DEBUG, "Sync demote and prefetch\n");
          sync_demote = true;
          sync_prefetch = true;
        }
        break;
      case 'n':
        sync_demote = false;
        break;
      case 'l':
        gLogLevel = atoi(optarg);
        break;
      case 't':
        total_requests = atol(optarg);
        break;
      case 'b':
        buf_size = atol(optarg);
        break;
      case 'i':
        iter = atoi(optarg);
        break;
      case 'g':
        noAcc = true;
        break;
      default:
        break;
    }
  }
  return 0;
}