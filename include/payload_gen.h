#ifndef PAYLOAD_GEN_H
#define PAYLOAD_GEN_H

#include <string>
#include <stdint.h>
using namespace std;
std::string gen_compressible_string(const char *append_string, int input_size);
char *gen_compressible_buf(const char *append_string, int input_size);

void random_permutation(uint64_t *array, int size);
void **create_random_chain_starting_at(int size, void **st_addr);

/*
  @return a random chain for pointer chasing of size bytes
*/
void **create_random_chain(int size);

#endif