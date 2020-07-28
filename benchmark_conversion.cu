/*
benchmark timing of different conversion implementation; 
testing performance only, not for correctness 
(will debug & provide correct output later if any candidate has high performance )
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime_api.h>
#define FP_TYPE float
#define POSIT_TYPE uint8_t
#include "posit_cuda.cuh"
texture<float, 1, cudaReadModeElementType> t_features;

__constant__ uint16_t table_[256];
__global__ void p2f(POSIT_TYPE in[], FP_TYPE out[], int n) {

   int i = blockDim.x * blockIdx.x + threadIdx.x;

   if (i < n) {
        fp16 temp = in[i] << 8;
        out [i] = fp16tofp32_gpu(temp);
   }
}  
__global__ void p2f_lookup(POSIT_TYPE in[], FP_TYPE out[], int n) {

   int i = blockDim.x * blockIdx.x + threadIdx.x;
   
   if (i < n) {
       
        uint32_t temp = table_[in[i]]<<16;
        out [i] = *((float*)&temp);
   }
}  
__global__ void p2f_lookup_shared_mem(POSIT_TYPE in[], FP_TYPE out[], int n) {

   int i = blockDim.x * blockIdx.x + threadIdx.x;
   
   if (i < n) {
   // prepare shared mem data
        __shared__ uint16_t table_shared[256];
        table_shared[threadIdx.x] = table_[threadIdx.x];
        __syncthreads();
        uint32_t temp = table_shared[in[i]]<<16;
        out [i] = *((float*)&temp);
   }
}  

__global__ void p2f_lookup_texture(POSIT_TYPE in[], FP_TYPE out[], int n) {

   int i = blockDim.x * blockIdx.x + threadIdx.x;
   
   if (i < n) {
   
        out [i]= tex1Dfetch(t_features,in[i]);
        //out [i] = *((float*)&temp);
   }
}  

__global__ void p2f_dummy_1op(FP_TYPE in[], FP_TYPE out[], int n) {

   int i = blockDim.x * blockIdx.x + threadIdx.x;
   
   if (i < n) {
   
        out [i]+= 0.2*in[i];
   }
}  
__global__ void p2f_dummy_coppy(FP_TYPE in[], FP_TYPE out[], int n) {

   int i = blockDim.x * blockIdx.x + threadIdx.x;
   
   if (i < n) {
   
        out [i] = in[i];
       
   }
}  
__global__ void p2f_dummy_coppy_uint8(POSIT_TYPE in[], POSIT_TYPE out[], int n) {

   int i = blockDim.x * blockIdx.x + threadIdx.x;
   
   if (i < n) {
   
        out [i] = in[i];
       
   }
}  
/* Host code */
int main(int argc, char* argv[]) {
   int n, i;
   POSIT_TYPE *h_in, *d_in;
   FP_TYPE *h_out, *d_out;
   
   FP_TYPE *table_global;
   
   int threads_per_block;
   int block_count;
   //size_t size;
	cudaEvent_t start, stop;
  float elapsedTime;
   /* Get number of components in vector */
   if (argc != 2) {
      fprintf(stderr, "usage: %s <vector order>\n", argv[0]);
      exit(0);
   }
   n = strtol(argv[1], NULL, 10);
     uint16_t lookup_table[256] = {0x0, 0x3380, 0x3580, 0x3680, 0x3780, 0x3800, 0x3880, 0x3900, 0x3980, 0x39c0, 0x3a00, 0x3a40, 0x3a80, 0x3ac0, 0x3b00, 0x3b40, 0x3b80, 0x3ba0, 0x3bc0, 0x3be0, 0x3c00, 0x3c20, 0x3c40, 0x3c60, 0x3c80, 0x3ca0, 0x3cc0, 0x3ce0, 0x3d00, 0x3d20, 0x3d40, 0x3d60, 0x3d80, 0x3d90, 0x3da0, 0x3db0, 0x3dc0, 0x3dd0, 0x3de0, 0x3df0, 0x3e00, 0x3e10, 0x3e20, 0x3e30, 0x3e40, 0x3e50, 0x3e60, 0x3e70, 0x3e80, 0x3e90, 0x3ea0, 0x3eb0, 0x3ec0, 0x3ed0, 0x3ee0, 0x3ef0, 0x3f00, 0x3f10, 0x3f20, 0x3f30, 0x3f40, 0x3f50, 0x3f60, 0x3f70, 0x3f80, 0x3f90, 0x3fa0, 0x3fb0, 0x3fc0, 0x3fd0, 0x3fe0, 0x3ff0, 0x4000, 0x4010, 0x4020, 0x4030, 0x4040, 0x4050, 0x4060, 0x4070, 0x4080, 0x4090, 0x40a0, 0x40b0, 0x40c0, 0x40d0, 0x40e0, 0x40f0, 0x4100, 0x4110, 0x4120, 0x4130, 0x4140, 0x4150, 0x4160, 0x4170, 0x4180, 0x41a0, 0x41c0, 0x41e0, 0x4200, 0x4220, 0x4240, 0x4260, 0x4280, 0x42a0, 0x42c0, 0x42e0, 0x4300, 0x4320, 0x4340, 0x4360, 0x4380, 0x43c0, 0x4400, 0x4440, 0x4480, 0x44c0, 0x4500, 0x4540, 0x4580, 0x4600, 0x4680, 0x4700, 0x4780, 0x4880, 0x4980, 0x4b80, 0xff80, 0xcb80, 0xc980, 0xc880, 0xc780, 0xc700, 0xc680, 0xc600, 0xc580, 0xc540, 0xc500, 0xc4c0, 0xc480, 0xc440, 0xc400, 0xc3c0, 0xc380, 0xc360, 0xc340, 0xc320, 0xc300, 0xc2e0, 0xc2c0, 0xc2a0, 0xc280, 0xc260, 0xc240, 0xc220, 0xc200, 0xc1e0, 0xc1c0, 0xc1a0, 0xc180, 0xc170, 0xc160, 0xc150, 0xc140, 0xc130, 0xc120, 0xc110, 0xc100, 0xc0f0, 0xc0e0, 0xc0d0, 0xc0c0, 0xc0b0, 0xc0a0, 0xc090, 0xc080, 0xc070, 0xc060, 0xc050, 0xc040, 0xc030, 0xc020, 0xc010, 0xc000, 0xbff0, 0xbfe0, 0xbfd0, 0xbfc0, 0xbfb0, 0xbfa0, 0xbf90, 0xbf80, 0xbf70, 0xbf60, 0xbf50, 0xbf40, 0xbf30, 0xbf20, 0xbf10, 0xbf00, 0xbef0, 0xbee0, 0xbed0, 0xbec0, 0xbeb0, 0xbea0, 0xbe90, 0xbe80, 0xbe70, 0xbe60, 0xbe50, 0xbe40, 0xbe30, 0xbe20, 0xbe10, 0xbe00, 0xbdf0, 0xbde0, 0xbdd0, 0xbdc0, 0xbdb0, 0xbda0, 0xbd90, 0xbd80, 0xbd60, 0xbd40, 0xbd20, 0xbd00, 0xbce0, 0xbcc0, 0xbca0, 0xbc80, 0xbc60, 0xbc40, 0xbc20, 0xbc00, 0xbbe0, 0xbbc0, 0xbba0, 0xbb80, 0xbb40, 0xbb00, 0xbac0, 0xba80, 0xba40, 0xba00, 0xb9c0, 0xb980, 0xb900, 0xb880, 0xb800, 0xb780, 0xb680, 0xb580, 0xb380 };
     
     cudaMemcpyToSymbol(table_, lookup_table, 256*sizeof(uint16_t));

     uint32_t lookup_table_int[256];
     for (int j =0; j<256; j ++)
         lookup_table_int[j] = lookup_table[j] <<16;

   /* Allocate input vectors in host memory */
   h_in = (POSIT_TYPE*) malloc(n*sizeof(POSIT_TYPE));
   h_out = (FP_TYPE*) malloc(n*sizeof(FP_TYPE));
   
    srand(0);   // Initialization, should only be called once.
  
   /* Initialize input vectors */
   for (i = 0; i < n; i++) {
	 
      h_in[i] = rand()%256;
   }



   /* Allocate vectors in device memory */
   cudaMalloc(&d_in, n*sizeof(POSIT_TYPE));
   cudaMalloc(&d_out, n*sizeof(FP_TYPE));
   cudaMalloc(&table_global, 256*sizeof(FP_TYPE));

   /* Copy vectors from host memory to device memory */
   cudaMemcpy(d_in, h_in, n*sizeof(POSIT_TYPE), cudaMemcpyHostToDevice);
   cudaMemcpy(table_global, lookup_table_int, 256*sizeof(FP_TYPE), cudaMemcpyHostToDevice);
   // this wont give the correct answer cuz fptype = 4 bytes, lookup_table 2 bytes per element, simply append 0x00 to each element in lookup_table to fix this
   //cudaMemcpy(d_out, h_out, n*sizeof(FP_TYPE), cudaMemcpyHostToDevice);

   /* Define block size */
   threads_per_block = 256;
   cudaDeviceSynchronize();
   block_count = (n + threads_per_block - 1)/threads_per_block;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
   
	cudaEventRecord(start,0);

   p2f<<<block_count, threads_per_block>>>(d_in, d_out, n);

   //cudaDeviceSynchronize();
	
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
      
  cudaEventElapsedTime(&elapsedTime, start,stop);
  cudaMemcpy(h_out, d_out, n*sizeof(FP_TYPE), cudaMemcpyDeviceToHost);
 
  printf("Elapsed time himeshi's implementation: %f ms\n" ,elapsedTime);
    
    //cudaDeviceSynchronize();
	cudaEventRecord(start,0);

   p2f_lookup<<<block_count, threads_per_block>>>(d_in, d_out, n);

   //cudaDeviceSynchronize();
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
  cudaMemcpy(h_out, d_out, n*sizeof(FP_TYPE), cudaMemcpyDeviceToHost);
  cudaEventElapsedTime(&elapsedTime, start,stop);
  printf("Elapsed time lookup constant mem: %f ms\n" ,elapsedTime);
  
   //cudaDeviceSynchronize();
	cudaEventRecord(start,0);

   p2f_lookup_shared_mem<<<block_count, threads_per_block>>>(d_in, d_out, n);

   //cudaDeviceSynchronize();
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
  cudaMemcpy(h_out, d_out, n*sizeof(FP_TYPE), cudaMemcpyDeviceToHost);
  cudaEventElapsedTime(&elapsedTime, start,stop);
  printf("Elapsed time lookup shared mem: %f ms\n" ,elapsedTime);
   
 
/* 
    for debugging
     for (int k = 0; k <16; k++)
         printf("%d %f\n", h_in[k], h_out[k]);
*/
   cudaChannelFormatDesc chDesc0 = cudaCreateChannelDesc<int>();
    t_features.filterMode = cudaFilterModePoint;
    t_features.normalized = false;
    t_features.channelDesc = chDesc0;

	if(cudaBindTexture(NULL, &t_features, table_global, &chDesc0, n*sizeof(uint32_t)) != CUDA_SUCCESS)
        printf("Couldn't bind features array to texture!\n");
    //cudaDeviceSynchronize();
	cudaEventRecord(start,0);

   p2f_lookup_texture<<<block_count, threads_per_block>>>(d_in, d_out, n);

   //cudaDeviceSynchronize();
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
  cudaMemcpy(h_out, d_out, n*sizeof(FP_TYPE), cudaMemcpyDeviceToHost);
  cudaEventElapsedTime(&elapsedTime, start,stop);
  printf("Elapsed time texture lookup: %f ms\n" ,elapsedTime);
     FP_TYPE *  d_out_dummy ; 
    cudaMalloc(&d_out_dummy, n*sizeof(FP_TYPE));
    //cudaDeviceSynchronize();
	cudaEventRecord(start,0);

   p2f_dummy_1op<<<block_count, threads_per_block>>>(d_out, d_out_dummy, n);

   //cudaDeviceSynchronize();
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
    
  cudaMemcpy(h_out, d_out, n*sizeof(FP_TYPE), cudaMemcpyDeviceToHost);
  cudaEventElapsedTime(&elapsedTime, start,stop);
  printf("Elapsed time 1 MAC float : %f ms\n" ,elapsedTime);
 
    // cudaDeviceSynchronize();
	cudaEventRecord(start,0);

   p2f_dummy_coppy<<<block_count, threads_per_block>>>(d_out, d_out_dummy, n);

   //cudaDeviceSynchronize();
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
  cudaMemcpy(h_out, d_out, n*sizeof(FP_TYPE), cudaMemcpyDeviceToHost);
  cudaEventElapsedTime(&elapsedTime, start,stop);
  printf("Elapsed time 1 copy fp32 : %f ms\n" ,elapsedTime);
 
      POSIT_TYPE *  d_in_dummy ; 
    cudaMalloc(&d_in_dummy, n*sizeof(POSIT_TYPE));
    // cudaDeviceSynchronize();
    
	cudaEventRecord(start,0);

   p2f_dummy_coppy_uint8<<<block_count, threads_per_block>>>(d_in, d_in_dummy, n);

  // cudaDeviceSynchronize();
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
  cudaMemcpy(h_out, d_out, n*sizeof(FP_TYPE), cudaMemcpyDeviceToHost);
  cudaEventElapsedTime(&elapsedTime, start,stop);
  printf("Elapsed time 1 copy uint_8 : %f ms\n" ,elapsedTime);
 
  
 
/*
  
      for (int k = 0; k <16; k++)
         printf("%d %f\n", h_in[k], h_out[k]);
*/
   /* Free device memory */
   cudaFree(d_out);
   cudaFree(d_in);
 

   /* Free host memory */
   free(h_out);
   free(h_in);

   return 0;
}  /* main */



