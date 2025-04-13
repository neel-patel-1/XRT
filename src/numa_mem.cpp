#include "numa_mem.h"
#include "print_utils.h"
#include <unistd.h>

uint64_t align(uint64_t v, uint64_t alignto)
{
	return  (v + alignto - 1) & ~(alignto - 1);
}


int set_mempolicy(int mode, const unsigned long *nodemask, unsigned long maxnode)
{
	return syscall(__NR_set_mempolicy, mode, nodemask, maxnode);
}

off_t file_sz(int fd)
{
	return lseek(fd, 0, SEEK_END);
}

int alloc_node_mem(uint64_t sz, int n, void **paddr, int pg_size)
{
  int fd;
  int rc;
  void *addr;
	uint64_t node_mask;

  if (sz == 0) {
		*paddr = 0;
		return 0;
	}

  if(pg_size == 4096){
    fd = memfd_create("temp", 0);
  }else{
    fd = memfd_create("temp", MFD_HUGETLB | MFD_HUGE_1GB);
  }
  if (fd < 0) {
    PRINT("Failed to create memfd\n");
    return -1;
  }

  LOG_PRINT(LOG_VERBOSE, "Size Before page alignment %lu\n", sz);
  LOG_PRINT(LOG_VERBOSE, "Allocating %lu bytes on node %d\n", align(sz, pg_size), n);

  rc = ftruncate(fd, align(sz, pg_size));
  if (rc) {
    PRINT("Failed to ftruncate %lu\n", sz);
    close(fd);
    return -1;
  }

  node_mask = 1ULL << n;
	rc = set_mempolicy(MPOL_BIND, &node_mask, 64);
	if (rc) {
		rc = -errno;
		PRINT("failed to bind memory range %s\n", strerror(errno));
		return rc;
	}


  *paddr = mmap(NULL, file_sz(fd), PROT_READ | PROT_WRITE,
		MAP_POPULATE | MAP_SHARED, fd, 0);
	close(fd);
	rc = set_mempolicy(MPOL_DEFAULT, NULL, 64);
	if (rc || *paddr == MAP_FAILED) {
		rc = -errno;
		if (*paddr != MAP_FAILED)
			munmap(*paddr, file_sz(fd));
		else{
			PRINT("Failed to mmap %lu from node %d\n",
        align(sz, pg_size), n);
      _exit(ENOMEM);
    }
		return rc;
	}

	return 0;

}

int alloc_numa_mem(struct numa_mem *nm, int pg_size, int node)
{
  int rc;

  void *addr;
  rc = alloc_node_mem(nm->size, node, &addr, pg_size);
  if(rc){
    PRINT("Failed to allocate memory\n");
    return -1;
  }

  nm->base_addr = addr;

  return 0;
}

void* alloc_offset(uint64_t sz, uint64_t *ptotal)
{
	void *p = (void *)(*ptotal);

	*ptotal += sz;
	*ptotal += 0xfff;
	*ptotal &= ~0xfffUL;

  LOG_PRINT(LOG_DEBUG,"Total allocated: %lu\n", *ptotal);

	return p;
}

void sub_offset(struct numa_mem *nm, uint64_t sub)
{
  nm->size -= sub;
}

void reserve_numa_mem(struct numa_mem *nm, uint64_t sz, int node)
{
  nm->size += sz;
}

uint64_t numa_base_addr(struct numa_mem *nm, int node)
{
	return (uint64_t)(nm->base_addr);
}

void add_base_addr(struct numa_mem *nm, void **ptr)
{
  LOG_PRINT(LOG_DEBUG,"Base address: %p\n", nm->base_addr);
  PTR_ADD(*ptr, numa_base_addr(nm, 0));
}

void * alloc_numa_offset(struct numa_mem *nm, uint64_t sz, uint32_t off){
  nm->size += off;
  return alloc_offset(sz, &nm->size);
}