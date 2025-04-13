#ifndef __NUMA_MEM_H__

#include <linux/memfd.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <stdint.h>
#include <sys/mman.h>
#include <linux/mempolicy.h>
#include <errno.h>
#include <string.h>


/* adapted from dsa-perf-micros: https://github.com/intel/dsa-perf-micros */

#define CACHE_LINE_SIZE 64
#define PTR(p) ((void *)(uintptr_t)(p))
#define PTR_ADD(p, a)	{ p = (void *)((uintptr_t)(p) + (uintptr_t)a); }

struct numa_mem {
  void *base_addr;
  uint64_t size;
};

uint64_t align(uint64_t v, uint64_t alignto);
int set_mempolicy(int mode, const unsigned long *nodemask, unsigned long maxnode);
off_t file_sz(int fd);
int alloc_node_mem(uint64_t sz, int n, void **paddr, int pg_size);
int alloc_numa_mem(struct numa_mem *nm, int pg_size, int node);
uint64_t numa_base_addr(struct numa_mem *nm, int node);
void add_base_addr(struct numa_mem *nm, void **ptr);
void *alloc_numa_offset(struct numa_mem *nm, uint64_t sz, uint32_t off);
void sub_offset(struct numa_mem *nm, uint64_t sub);
void reserve_numa_mem(struct numa_mem *nm, uint64_t sz, int node);

#endif