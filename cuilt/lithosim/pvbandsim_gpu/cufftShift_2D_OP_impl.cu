/*********************************************************************
 * Copyright © 2011-2014,
 * Marwan Abdellah: <abdellah.marwan@gmail.com>
 *
 * This library (cufftShift) is free software; you can redistribute it
 * and/or modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 ********************************************************************/

#ifndef CUFFTSHIFT_2D_IMPL_CU
#define CUFFTSHIFT_2D_IMPL_CU

// #include "lithosim/pvbandsim_gpu/cufftshift2d/configGPU.h"
#include "lithosim/pvbandsim_gpu/cufftShiftShared.h"
#include "lithosim/pvbandsim_gpu/cufftShift_2D_OP.cu"
//#include "cufftShiftShared.h"
//#include "cufftShift_2D_OP.cu"

// template <typename T>
// extern
kernelConf* GenAutoConf_2D(int N)
{
    kernelConf* autoConf = (kernelConf*) malloc(sizeof(kernelConf));

    int threadsPerBlock_X;
    int threadsPerBlock_Y;

    if (2 <= N && N < 4)
    {
        threadsPerBlock_X = 2;
        threadsPerBlock_Y = 2;
    }
    if (4 <= N && N < 8)
    {
        threadsPerBlock_X = 4;
        threadsPerBlock_Y = 4;
    }
    if (8 <= N && N < 16)
    {
        threadsPerBlock_X = 8;
        threadsPerBlock_Y = 8;
    }
    if (16 <= N && N < 32)
    {
        threadsPerBlock_X = 16;
        threadsPerBlock_Y = 16;
    }
    if (N >= 32)
    {
        threadsPerBlock_X = 16;
        threadsPerBlock_Y = 16;
    }

    autoConf->block = dim3(threadsPerBlock_X, threadsPerBlock_Y, 1);
    autoConf->grid = dim3((N / threadsPerBlock_X), (N / threadsPerBlock_Y), 1);

    //printf("Auto Block Conf [%d]x[%d] \n", autoConf->block.x, autoConf->block.y);
    //printf("Auto Grid Conf [%d]x[%d] \n", autoConf->grid.x, autoConf->grid.y);

    return autoConf;
}


// __global__ void cufftShift_2D_impl(cufftComplex* input, cufftComplex* output, int NX, int NY);
void cufftShift_2D_impl(cufftComplex* input, cufftComplex* output, int NX, int NY)
{
    if (NX == NY)
    {
        const int N = NX;
        kernelConf* conf = GenAutoConf_2D(N);
        cufftShift_2D_kernel <<< conf->grid, conf->block >>> (input, output, N);
        free(conf);
    }
    else
    {
        printf("The library is supporting NxN arrays only \n");
        exit(0);
    }
}

// template <typename T>
// __global__ void cufftShift_2D_config_impl(cufftComplex* input, cufftComplex* output, int NX, int NY, kernelConf* conf);
// void cufftShift_2D_config_impl(cufftComplex* input, cufftComplex* output, int NX, int NY, kernelConf* conf)
// {
//     if (NX == NY)
//     {
//         const int N = NX;
//         cufftShift_2D_kernel <<< conf->grid, conf->block >>> (input, output, N);
//     }

//     else
//     {
//         printf("The library is supporting NxN arrays only \n");
//         exit(0);
//     }
// }

// template void cufftShift_2D_impl <cufftReal>
// (cufftReal* input, cufftReal* output, int NX, int NY);

// template void cufftShift_2D_impl <cufftDoubleReal>
// (cufftDoubleReal* input, cufftDoubleReal* output, int NX, int NY);

// extern "C" void cufftShift_2D_impl(cufftComplex* input, cufftComplex* output, int NX, int NY);

// template void cufftShift_2D_impl <cufftComplex>
// (cufftComplex* input, cufftComplex* output, int NX, int NY);

// template void cufftShift_2D_impl <cufftDoubleComplex>
// (cufftDoubleComplex* input, cufftDoubleComplex* output, int NX, int NY);

// template void cufftShift_2D_config_impl <cufftReal>
// (cufftReal* input, cufftReal* output, int NX, int NY, kernelConf* conf);

// template void cufftShift_2D_config_impl <cufftDoubleReal>
// (cufftDoubleReal* input, cufftDoubleReal* output, int NX, int NY, kernelConf* conf);

// template void cufftShift_2D_config_impl <cufftComplex>
// (cufftComplex* input, cufftComplex* output, int NX, int NY, kernelConf* conf);

// template void cufftShift_2D_config_impl <cufftDoubleComplex>
// (cufftDoubleComplex* input, cufftDoubleComplex* output, int NX, int NY, kernelConf* conf);

#endif // CUFFTSHIFT_2D_IMPL_CU
