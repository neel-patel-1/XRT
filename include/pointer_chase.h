#ifndef P_CHASE
#define P_CHASE
extern volatile void *chase_pointers_global;
void chase_pointers(void **memory, int count);
void debug_chain(void **memory);
#endif