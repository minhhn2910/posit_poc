#### Benchmarking conversion operation in ms
compile : 

`nvcc benchmark_conversion.cu -O3 `

run : 
`./a.out 500000000`

Should use CUDA_VISIBLE_DEVICES flag to choose free GPUs (not occupied by other processes)
#### Benchmarking conversion operation in cycle
compile : 

`nvcc benchmark_conversion_cycle.cu `

run : 
`./a.out 256` 

Launch 1 block only to avoid thread waiting for scheduling. Should not use -O3 cuz it will do some unpredictable optimization (shuffle the order of the assemby instructions) affecting the cycle counts.

#### Benchmarking MAC (multiply accumulate) operation
compile : 

`nvcc benchmark_mac.cu -O3 `

run : 
`./a.out 50000000` 


