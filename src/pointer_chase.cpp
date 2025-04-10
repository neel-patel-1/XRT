#include "pointer_chase.h"
#include "print_utils.h"
volatile void *chase_pointers_global;
void chase_pointers(void **memory, int count){
  void ** p = (void **)memory;
  while (count -- > 0) {
    LOG_PRINT( LOG_TOO_VERBOSE, "  %p -> %p\n", p, *p);
    p = (void **) *p;
  }
  chase_pointers_global = *p;
  if(count > 0){
    LOG_PRINT( LOG_TOO_VERBOSE, "Remaining count: %d\n", count);
  }
}

void debug_chain(void **memory){
  void ** p = memory;
  size_t count = 0;
  LOG_PRINT( LOG_TOO_VERBOSE, "chain at %p:\n", memory);
  do {
    LOG_PRINT( LOG_TOO_VERBOSE, "  %p -> %p\n", p, *p);
    p = (void **) *p;
    count++;
  } while (p != memory);
  LOG_PRINT( LOG_TOO_VERBOSE, "# of pointers in chain: %lu\n", count);


}