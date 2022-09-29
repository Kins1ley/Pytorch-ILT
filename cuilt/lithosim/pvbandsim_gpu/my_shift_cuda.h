#ifndef MY_SHIFT_CUDA_H
#define MY_SHIFT_CUDA_H

#include "lithosim/pvbandsim_gpu/kiss_fft.h"
#include "lithosim/pvbandsim_gpu/cufftShift.h"

//#ifdef __cplusplus
//extern "C" {
//#endif

typedef struct kiss_fftnd_state * kiss_fftnd_cfg;

void my_shift_cuda(const kiss_fft_cpx *fin,kiss_fft_cpx *fout);
//
//#ifdef __cplusplus
//}
//#endif
#endif
