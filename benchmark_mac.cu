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

#define N_LOOP 40
__constant__ uint32_t table_[64*256];

__global__ void mac_lookup(POSIT_TYPE in1[],POSIT_TYPE in2[], FP_TYPE out[], int n) {

   int i = blockDim.x * blockIdx.x + threadIdx.x;
   
   if (i < n - N_LOOP) {
        short index = ((in1[i] & 0x7f ) << 7 )|| ((in2[i]& 0x7f )) ; 
        uint32_t temp = table_[index];
        for (int k =0;k < N_LOOP; k++){
             index = ((in1[i+k] & 0x7f ) << 7 )|| ((in2[i+k]& 0x7f)) ; 
             temp += table_[index];
         }
        out [i] = *((float*)&temp);
        
   }
}  


__global__ void mac_lookup_texture(POSIT_TYPE in1[],POSIT_TYPE in2[], FP_TYPE out[], int n) {

   int i = blockDim.x * blockIdx.x + threadIdx.x;
   
   if (i < n - N_LOOP) {
        short index = ((in1[i] & 0x7f ) << 7 )|| ((in2[i]& 0x7f )) ; 
        int temp = tex1Dfetch(t_features,index);
        for (int k =0;k < N_LOOP; k++){
             index = ((in1[i+k] & 0x7f ) << 7 )|| ((in2[i+k]& 0x7f ) ) ; 
             temp += tex1Dfetch(t_features,index);
         }
         out [i] = *((float*)&temp);
      
   }
}  

__global__ void mac_fp32(FP_TYPE in1[],FP_TYPE in2[], FP_TYPE out[], int n) {

   int i = blockDim.x * blockIdx.x + threadIdx.x;
   
   if (i < n - N_LOOP) {
         float temp = in1[i]*in2[i];
         for (int k = 0; k < N_LOOP; k++){
             
             temp += in1[i+k]*in2[i+k];
         }
         out [i] = temp;
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
   POSIT_TYPE *h_in1, *h_in2 , *d_in1, *d_in2;
   FP_TYPE *h_out, *h_in_float, *d_out, *d_in1_float,*d_in2_float;
   
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
     

     uint32_t lookup_table_constant[64*256]; // just small array to fit to  the constant mem. not for correctness
     
        for (i = 0; i < 64*256; i++) {

          lookup_table_constant[i] = rand();
       }
    cudaMemcpyToSymbol(table_, lookup_table_constant, 256*64*sizeof(uint32_t));

   /* Allocate input vectors in host memory */
   h_in1 = (POSIT_TYPE*) malloc(n*sizeof(POSIT_TYPE));
   h_in2 = (POSIT_TYPE*) malloc(n*sizeof(POSIT_TYPE));
   h_in_float = (FP_TYPE*) malloc(n*sizeof(FP_TYPE));
   h_out = (FP_TYPE*) malloc(n*sizeof(FP_TYPE));
   
    srand(0);   // Initialization, should only be called once.
  
   /* Initialize input vectors */
   for (i = 0; i < n; i++) {
	 
      h_in1[i] = rand()%256;
      h_in2[i] = rand()%256;
      h_in_float[i] = h_in1[i];
   }



   /* Allocate vectors in device memory */
   cudaMalloc(&d_in1, n*sizeof(POSIT_TYPE));
   cudaMalloc(&d_in2, n*sizeof(POSIT_TYPE));
   cudaMalloc(&d_in1_float, n*sizeof(FP_TYPE));
   cudaMalloc(&d_in2_float, n*sizeof(FP_TYPE));
   cudaMalloc(&d_out, n*sizeof(FP_TYPE));
   cudaMalloc(&table_global, 64*256*sizeof(FP_TYPE)); //for texture;

   /* Copy vectors from host memory to device memory */
   cudaMemcpy(d_in1, h_in1, n*sizeof(POSIT_TYPE), cudaMemcpyHostToDevice);
   cudaMemcpy(d_in2, h_in2, n*sizeof(POSIT_TYPE), cudaMemcpyHostToDevice);
   cudaMemcpy(d_in1_float, h_in_float, n*sizeof(FP_TYPE), cudaMemcpyHostToDevice);
   cudaMemcpy(d_in2_float, h_in_float, n*sizeof(FP_TYPE), cudaMemcpyHostToDevice);
   
   cudaMemcpy(table_global, lookup_table_constant, 64*256*sizeof(FP_TYPE), cudaMemcpyHostToDevice);
   //cudaMemcpy(d_out, h_out, n*sizeof(FP_TYPE), cudaMemcpyHostToDevice);
 
      POSIT_TYPE *  d_in_dummy ; 
    cudaMalloc(&d_in_dummy, n*sizeof(POSIT_TYPE)); 
   cudaChannelFormatDesc chDesc0 = cudaCreateChannelDesc<int>();
    t_features.filterMode = cudaFilterModePoint;
    t_features.normalized = false;
    t_features.channelDesc = chDesc0;

	if(cudaBindTexture(NULL, &t_features, table_global, &chDesc0, n*sizeof(uint32_t)) != CUDA_SUCCESS)
        printf("Couldn't bind features array to texture!\n");
    //cudaDeviceSynchronize();
     cudaDeviceSynchronize();

      // check for error
      cudaError_t error = cudaGetLastError();
      if(error != cudaSuccess)
      {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
      } 

    FP_TYPE *  d_out_dummy ; 
    cudaMalloc(&d_out_dummy, n*sizeof(FP_TYPE));
    cudaDeviceSynchronize();
    
   /* Define block size */
   threads_per_block = 256;

   block_count = (n + threads_per_block - 1)/threads_per_block;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
	cudaEventRecord(start,0);

   mac_lookup<<<block_count, threads_per_block>>>(d_in1,d_in2, d_out, n);

   //cudaDeviceSynchronize();
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
  cudaMemcpy(h_out, d_out, n*sizeof(FP_TYPE), cudaMemcpyDeviceToHost);
  //elapsedTime = 0;
  cudaEventElapsedTime(&elapsedTime, start,stop);
  printf("Elapsed time lookup constant mem: %f ms\n" ,elapsedTime);
  //exit(0);

/* 
    for debugging
     for (int k = 0; k <16; k++)
         printf("%d %f\n", h_in[k], h_out[k]);
*/
  //   cudaDeviceSynchronize();
  
	cudaEventRecord(start,0);

   mac_lookup_texture<<<block_count, threads_per_block>>>(d_in1,d_in2, d_out, n);

   //cudaDeviceSynchronize();
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
  cudaMemcpy(h_out, d_out, n*sizeof(FP_TYPE), cudaMemcpyDeviceToHost);
  elapsedTime = 0;
  cudaEventElapsedTime(&elapsedTime, start,stop);
  printf("Elapsed time texture lookup: %f ms\n" ,elapsedTime);
 
    //cudaDeviceSynchronize();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start,0);
     
 
   mac_fp32<<<block_count, threads_per_block>>>(d_in1_float, d_in2_float, d_out, n);

   //cudaDeviceSynchronize();
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
    elapsedTime = 0;
  cudaMemcpy(h_out, d_out, n*sizeof(FP_TYPE), cudaMemcpyDeviceToHost);
  cudaEventElapsedTime(&elapsedTime, start,stop);
  printf("Elapsed time 1 MAC float : %f ms\n" ,elapsedTime);
 //exit(0);
   //  cudaDeviceSynchronize();
	cudaEventRecord(start,0);

   p2f_dummy_coppy<<<block_count, threads_per_block>>>(d_out, d_out_dummy, n);

  // cudaDeviceSynchronize();
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
  cudaMemcpy(h_out, d_out, n*sizeof(FP_TYPE), cudaMemcpyDeviceToHost);
  cudaEventElapsedTime(&elapsedTime, start,stop);
  printf("Elapsed time 1 copy fp32 : %f ms\n" ,elapsedTime);

   // cudaDeviceSynchronize();
	cudaEventRecord(start,0);

   p2f_dummy_coppy_uint8<<<block_count, threads_per_block>>>(d_in1, d_in_dummy, n);

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
   //cudaFree(d_in);
 

   /* Free host memory */
   free(h_out);
   //free(h_in);

   return 0;
}  /* main */



