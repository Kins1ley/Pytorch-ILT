#ifndef KISS_FFTND_CUDA_H
#define KISS_FFTND_CUDA_H

#include "kiss_fft.h"

//#ifdef __cplusplus
//extern "C" {
//#endif

typedef struct kiss_fftnd_state * kiss_fftnd_cfg;

void kiss_fftnd_cuda(const kiss_fft_cpx *fin,kiss_fft_cpx *fout, int isinverse);

//#ifdef __cplusplus
//}
//#endif
#endif
