#include "payload_gen.h"
#include <random>
#include <cstring>
#include <ctime>
std::string gen_compressible_string(const char *append_string, int input_size){
  std::string payload = "";
  while(payload.size() < input_size){
    payload.append(append_string);
  }
  return payload;
}

char *gen_compressible_buf(const char *append_string, int input_size){
  char *payload = (char *)aligned_alloc(32, input_size);
  int append_strlen = strlen(append_string);
  int cpylen = append_strlen;
  int i=0;
  while(i < input_size){
    if(i + append_strlen > input_size){
      cpylen = input_size - i;
    }
    memcpy(&payload[i], append_string, cpylen);
    i+=cpylen;
  }
  return payload;
}

void random_permutation(uint64_t *array, int size){
  srand(time(NULL));

  for (int i = size - 1; i > 0; i--) {
      // Get a random index from 0 to i
      int j = rand() % (i + 1);

      // Swap array[i] with the element at random index
      uint64_t temp = array[i];
      array[i] = array[j];
      array[j] = temp;
  }
}

void **create_random_chain(int size){
  uint64_t len = size / 64;

  void ** memory = (void **)malloc(size);
  uint64_t  *indices = (uint64_t *)malloc(sizeof(uint64_t) * len);
  for (int i = 0; i < len; i++) {
    indices[i] = i;
  }
  random_permutation(indices, len);

  /* the memaddr is 8 bytes -- only read each cache line once */
  for (int i = 1; i < len; ++i) {
    memory[indices[i-1] * 8] = (void *) &memory[indices[i] * 8];
  }
  memory[indices[len - 1] * 8] = (void *) &memory[indices[0] * 8 ];
  return memory;
}

/*
st_addr: is an existing buffer of size bytes
return: a buffer containing addresses within the buffer starting at st_addr
* returns a buffer to be used as the source buffer for copies to a destination buffer that will be passed to
chase_pointers
*/

void **create_random_chain_starting_at(int size, void **st_addr){ /* only touches each cacheline*/
  uint64_t len = size / 64;
  void ** memory = (void **)malloc(size);
  uint64_t  *indices = (uint64_t *)malloc(sizeof(uint64_t) * len);
  for (int i = 0; i < len; i++) {
    indices[i] = i;
  }
  random_permutation(indices, len); /* have a random permutation of cache lines to pick*/

  for (int i = 1; i < len; ++i) {
    memory[indices[i-1] * 8] = (void *) &st_addr[indices[i] * 8];
  }
  memory[indices[len - 1] * 8] = (void *) &st_addr[indices[0] * 8 ];
  return memory; /* Why x8 ?*/

}