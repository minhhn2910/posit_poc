compile : 

`nvcc benchmark_conversion.cu -O3 `

run : 
`./a.out 500000000`

Should use CUDA_VISIBLE_DEVICES flag to choose free GPUs (not occupied by other processes)