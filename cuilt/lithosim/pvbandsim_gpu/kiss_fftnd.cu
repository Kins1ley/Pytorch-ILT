#include <cuda_runtime.h>
#include <cufft.h>          //CUFFT文件头

// Helper functions for CUDA
#include "device_launch_parameters.h"

#include <stdio.h>
#include "lithosim/pvbandsim_gpu/kiss_fft.h"
//#include "kiss_fft.h"

#define NX 2048
#define NY 2048



void kiss_fftnd_cuda(const kiss_fft_cpx *fin, kiss_fft_cpx *fout,int isinverse){
    cufftComplex *idata, *odata;   //显存数据指针
    //在显存中分配空间
    cudaMalloc((void**)&idata, sizeof(cufftComplex)*NX*NY);
    cudaMalloc((void**)&odata, sizeof(cufftComplex)*NX*NY);

    if(cudaGetLastError() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to allocate\n");
        return;
    }
    cudaMemcpy(idata, fin, sizeof(cufftComplex)*NX*NY, cudaMemcpyHostToDevice);
    /*
    Do not use for loop to assign the value
    */
    //创建CUFFT句柄
    cufftHandle plan;
    cufftPlan2d(&plan, NX, NY, CUFFT_C2C);

    //执行CUFFT
    if(isinverse){
        // printf("now excute the inverse fft \n");
        cufftExecC2C(plan, idata, odata, CUFFT_INVERSE);  //快速傅里叶反向变换
    }else{
        // printf("now excute the forward fft \n");
        cufftExecC2C(plan, idata, odata, CUFFT_FORWARD);  //快速傅里叶正变换
    }

    /*
    Still, Do not use for loop to assign the value.
    Use cudaMemcpy
    */
    cudaMemcpy(fout, odata, sizeof(cufftComplex)*NX*NY, cudaMemcpyDeviceToHost);
    //销毁句柄，并释放空间
    cufftDestroy(plan);
    cudaFree(idata);
    cudaFree(odata);
}


