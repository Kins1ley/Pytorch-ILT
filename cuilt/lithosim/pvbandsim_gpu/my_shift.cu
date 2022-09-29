#include <cuda_runtime.h>
#include <cufft.h>          //CUFFT文件头

// Helper functions for CUDA
#include "device_launch_parameters.h"

#include <stdio.h>
#include "lithosim/pvbandsim_gpu/kiss_fft.h"
#include "lithosim/pvbandsim_gpu/cufftShift.h"
//#include "kiss_fft.h"
//#include "cufftShift.h"

#define NX 2048
#define NY 2048

void my_shift_cuda(const kiss_fft_cpx *fin, kiss_fft_cpx *fout){
    cufftComplex *idata, *odata;   //显存数据指针
    //在显存中分配空间
    cudaMalloc((void**)&idata, sizeof(cufftComplex)*NX*NY);
    cudaMalloc((void**)&odata, sizeof(cufftComplex)*NX*NY);
    if(cudaGetLastError() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to allocate\n");
        return;
    }
    cudaMemcpy(idata, fin, sizeof(cufftComplex)*NX*NY, cudaMemcpyHostToDevice);
    cufftShift_2D_impl(idata, odata, NX, NY);
    cudaMemcpy(fout, odata, sizeof(cufftComplex)*NX*NY, cudaMemcpyDeviceToHost);
    //销毁句柄，并释放空间
    cudaFree(idata);
    cudaFree(odata);
}


