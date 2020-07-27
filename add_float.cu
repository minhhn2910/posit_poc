#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <device_functions.h>
#define FP_TYPE float
/* Kernel for vector addition */
static __device__ __inline__ int __shortadd2(const int value_a, const int value_b)
 {
  int ret;
  asm("{vadd2.s32.s32.s32.sat  %0, %1, %2, %0;}" : "=r"(ret) : "r"(value_a) , "r"(value_b));
 return ret;
 }
__global__ void Vec_add(FP_TYPE x[], FP_TYPE y[], FP_TYPE z[], int n) {
   /* blockDim.x = threads_per_block                            */
   /* First block gets first threads_per_block components.      */
   /* Second block gets next threads_per_block components, etc. */
   int i = blockDim.x * blockIdx.x + threadIdx.x;

   /* block_count*threads_per_block may be >= n */
   if (i < n) {
	   //~ for (int j = 0; j<1000; j++)
	   float x_val = x[i]; 
	   float y_val = y[i];
	   float z_val;
	   #pragma unroll 1
	   for (int k = 0; k<10000; k++)
		{	//z_val = __shortadd2(x_val,y_val);
			if(i != -2)
				z_val = x_val + y_val;		
		}
	// z_val  = __vadd2(x_val, y_val);
			//~ asm("{vadd2.s32.s32.s32.sat  %0, %1, %2, %0;}" : "=r"(z_val) : "r"(x_val) , "r"(y_val));
	
	z[i] = z_val;
   }
}  /* Vec_add */


/* Host code */
int main(int argc, char* argv[]) {
   int n, i;
   FP_TYPE *h_x, *h_y, *h_z;
   FP_TYPE *d_x, *d_y, *d_z;
   int threads_per_block;
   int block_count;
   size_t size;
	cudaEvent_t start, stop;
  float elapsedTime;
   /* Get number of components in vector */
   if (argc != 2) {
      fprintf(stderr, "usage: %s <vector order>\n", argv[0]);
      exit(0);
   }
   n = strtol(argv[1], NULL, 10);
   size = n*sizeof(FP_TYPE);

   /* Allocate input vectors in host memory */
   h_x = (FP_TYPE*) malloc(size);
   h_y = (FP_TYPE*) malloc(size);
   h_z = (FP_TYPE*) malloc(size);

   /* Initialize input vectors */
   for (i = 0; i < n; i++) {
	   int k= rand();
      h_x[i] = (i+1)%50;
      h_y[i] = -((n-k)%5);
   }

   //~ printf("h_x = ");
   //~ for (i = 0; i < n; i++)
      //~ printf("%d ", h_x[i]);
   //~ printf("\n");

   //~ printf("h_y = ");
   //~ for (i = 0; i < n; i++)
      //~ printf("%d ", h_y[i]);
   //~ printf("\n\n");

   /* Allocate vectors in device memory */
   cudaMalloc(&d_x, size);
   cudaMalloc(&d_y, size);
   cudaMalloc(&d_z, size);

   /* Copy vectors from host memory to device memory */
   cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

   /* Define block size */
   threads_per_block = 256;

   block_count = (n + threads_per_block - 1)/threads_per_block;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);

   Vec_add<<<block_count, threads_per_block>>>(d_x, d_y, d_z, n);

   cudaThreadSynchronize();
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
 cudaEventElapsedTime(&elapsedTime, start,stop);
 printf("Elapsed time : %f ms\n" ,elapsedTime);
  cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost);

   double avg_sum = 0.0;
   for (i = 0; i < n; i++)
      avg_sum = avg_sum + h_z[i];
      
  avg_sum/= n;
   
printf("%.3lf \n",avg_sum );
   /* Free device memory */
   cudaFree(d_x);
   cudaFree(d_y);
   cudaFree(d_z);

   /* Free host memory */
   free(h_x);
   free(h_y);
   free(h_z);

   return 0;
}  /* main */



