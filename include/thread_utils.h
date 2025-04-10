#ifndef _THREAD_UTILS
#define _THREAD_UTILS

#include <pthread.h>

int create_thread_pinned(pthread_t *thread, void* (*func)(void*), void *arg, int core_id);

#endif