# How to run

## After acquiring the repository: 

### For CUDA
from the parent folder:
you can run ./benchmark_cuda to run the executable file of the benchmark_cuda.cu file.

for the experiment, I used the following command to produce the executable file "benchmark_cuda":
* nvcc -O3 -arch=sm_70 benchmark_cuda.cu -o benchmark_cuda

### For the CPU baseline
from the parent folder FinalProject, cd into build.

Then use the following commands:

* rm -rf *
* cmake ..
* cmake --build .
* ./benchmark_native

The benchmark file was originally intended to use the tick library itself; instead, to simplify the downloading and testing of the file, the file was made to be self-contained so that it does not depend on the tick library.

The tick library contains 'tick/lib/cpp/hawkes/model/base/model_hawkes_loglik_single.cpp', which contains the ModelHawkesLogLikSingle::hessian_i function. This is the function that this project is trying to port to the GPU.

