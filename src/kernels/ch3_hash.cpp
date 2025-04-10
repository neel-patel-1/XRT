#include "ch3_hash.h"
uint64_t murmur_rehash_64A(uint64_t k) {
  const uint64_t m = 0xc6a4a7935bd1e995ULL;
  const int r = 47;

  uint64_t h = (uint64_t)SEED ^ (sizeof(uint64_t) * m);

  k *= m;
  k ^= k >> r;
  k *= m;

  h ^= k;
  h *= m;

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}

uint64_t
murmur_hash_64A(const void* const key, const size_t len, const uint32_t seed)
 {
  const uint64_t m = 0xc6a4a7935bd1e995ULL;
  const int r = 47;

  uint64_t h = seed ^ (len * m);

  const uint64_t* data = (const uint64_t*)key;
  const uint64_t* end = data + (len / 8);

  while (data != end) {
    uint64_t k = *data++;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }
  const uint8_t* data2 = (const uint8_t*)data;

  switch (len & 7) {
    case 7:
      h ^= (uint64_t)data2[6] << 48;
      __attribute__((fallthrough));
    case 6:
      h ^= (uint64_t)data2[5] << 40;
      __attribute__((fallthrough));
    case 5:
      h ^= (uint64_t)data2[4] << 32;
      __attribute__((fallthrough));
    case 4:
      h ^= (uint64_t)data2[3] << 24;
      __attribute__((fallthrough));
    case 3:
      h ^= (uint64_t)data2[2] << 16;
      __attribute__((fallthrough));
    case 2:
      h ^= (uint64_t)data2[1] << 8;
      __attribute__((fallthrough));
    case 1:
      h ^= (uint64_t)data2[0];
      h *= m;
  }

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}


uint32_t furc_get_bit(
    const char* const key,
    const size_t len,
    const uint32_t idx,
    uint64_t* hash,
    int32_t* old_ord_p) {
  int32_t ord = (idx >> 6);
  int n;

  if (key == NULL) {
    *old_ord_p = -1;
    return 0;
  }

  if (*old_ord_p < ord) {
    for (n = *old_ord_p + 1; n <= ord; n++) {
      hash[n] =
          ((n == 0) ? murmur_hash_64A(key, len, SEED)
                    : murmur_rehash_64A(hash[n - 1]));
    }
    *old_ord_p = ord;
  }

  return (hash[ord] >> (idx & 0x3f)) & 0x1;
}

uint32_t furc_hash(const char* const key, const size_t len, const uint32_t m) {
  uint32_t attempt;
  uint32_t d;
  uint32_t num;
  uint32_t i;
  uint32_t a;
  uint64_t hash[FURC_CACHE_SIZE];
  int32_t old_ord;

  // There are (ab)users of this function that pass in larger
  // numbers, and depend on the behavior not changing (ie we can't
  // just clamp to the max). Just let it go for now.

  // assert(m <= furc_maximum_pool_size());

  if (m <= 1) {
    return 0;
  }

  furc_get_bit(NULL, 0, 0, hash, &old_ord);
  for (d = 0; m > (1ul << d); d++)
    ;

  a = d;
  for (attempt = 0; attempt < MAX_TRIES; attempt++) {
    while (!furc_get_bit(key, len, a, hash, &old_ord)) {
      if (--d == 0) {
        return 0;
      }
      a = d;
    }
    a += FURC_SHIFT;
    num = 1;
    for (i = 0; i < d - 1; i++) {
      num = (num << 1) | furc_get_bit(key, len, a, hash, &old_ord);
      a += FURC_SHIFT;
    }
    if (num < m) {
      return num;
    }
  }

  // Give up; return 0, which is a legal value in all cases.
  return 0;
}
