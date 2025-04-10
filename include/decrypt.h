#ifndef DECRYPT_H
#define DECRYPT_H

#include "ippcp.h"

extern __thread IppsAES_GCMState *pState;

void gen_encrypted_feature(int payload_size, void **p_msgbuf, int *outsize);

static void decrypt_feature(void *cipher_inp, void *plain_out, int input_size, int *output_size);

#include "inline/decrypt.ipp"

#endif // DECRYPT_Hs