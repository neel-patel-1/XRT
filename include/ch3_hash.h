#ifndef HASH_H
#define HASH_H

#include <stdint.h>
#include <cstddef>
/* hash.h*/
/* Seed constant for MurmurHash64A selected by search for optimum diffusion
 * including recursive application.
 */
#define SEED 4193360111ul

/* Maximum number tries for in-range result before just returning 0. */
#define MAX_TRIES 32

/* Gap in bit index per attempt; limits us to 2^FURC_SHIFT shards.  Making this
 * larger will sacrifice a modest amount of performance.
 */
#define FURC_SHIFT 23

/* Size of cache for hash values; should be > MAXTRIES * (FURCSHIFT + 1) */
#define FURC_CACHE_SIZE 1024


/* MurmurHash64A performance-optimized for hash of uint64_t keys and seed = M0
 */
uint64_t murmur_rehash_64A(uint64_t k);
uint64_t murmur_hash_64A(const void* const key, const size_t len, const uint32_t seed);

uint32_t furc_get_bit(
    const char* const key,
    const size_t len,
    const uint32_t idx,
    uint64_t* hash,
    int32_t* old_ord_p);

uint32_t furc_hash(const char* const key, const size_t len, const uint32_t m);

#endif