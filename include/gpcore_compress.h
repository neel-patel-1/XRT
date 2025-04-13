#ifndef GPCORE_COMPRESS_H
#define GPCORE_COMPRESS_H

extern "C" {
  #include <zlib.h>
  #include <stdio.h>
  #include <stdint.h>
  #include <string.h>
}
#include "print_utils.h"

static int gpcore_do_decompress(void *dst, void *src, uInt src_len, uLong *out_len);
static void gpcore_do_deflate_decompress(void *dst, void *src, int src_len, int *out_len);
static inline uLong get_compress_bound(int payload_size);

#include "inline/gpcore_compress.ipp"

#endif