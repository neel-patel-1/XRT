#ifndef RACER_OPTS_H
#define RACER_OPTS_H

#include <stdint.h>
#include "print_utils.h"
extern bool sync_prefetch;
extern bool sync_demote;
extern uint64_t total_requests;
extern uint64_t buf_size;
extern uint64_t iter;
extern int gLogLevel;
extern bool noAcc;

int get_opts(int argc, char **argv);
#endif