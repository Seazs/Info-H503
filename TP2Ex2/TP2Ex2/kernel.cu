
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <iostream>


bool checkCuda(int* out_cpu, int* out_gpu, int N);

__global__ void addKernel(int* c, const int* a, const int* b, const int N)
{
    //TODO Change the kernel to work with bigger vectors
    const unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= N)
        return;
    c[i] = a[i] + b[i];
}

int main()
{
    //TODO Change arraySize to work with bigger vectors
    const int arraySize = 1000;
    int* a = (int*)malloc(arraySize * sizeof(int));
    int* b = (int*)malloc(arraySize * sizeof(int));

    for (int i = 0; i < arraySize; i++) {
        a[i] = i;
        b[i] = arraySize - i;
    }
    int* c = (int*)malloc(arraySize * sizeof(int));
    int* cpu_result = (int*)malloc(arraySize * sizeof(int));

    //1. Do the computation on the CPU and time it
    std::chrono::steady_clock::time_point start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < arraySize; i++) {
        cpu_result[i] = a[i] + b[i];
    }
    std::chrono::steady_clock::time_point stop_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_runtime_us = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu).count();


    //2. Do the computation on GPU and time it

    // Define the variable we need 
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;

    cudaError_t cudaStatus;
    cudaEvent_t start_gpu, stop_gpu; //cudaEvent are used to time the kernel
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // Define the size of the grid (block_size = #blocks in the grid)
    //and the size of a block (thread_size = #threads in a block)
    //TODO Change how block_size and thread_size are defined to work with bigger vectors 
    size_t N_threads = 1024;
    dim3 block_size((arraySize + (N_threads-1))/N_threads);
    dim3 thread_size(N_threads);
    

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element 
    //CudaEventRecord used to time the kernel
    cudaEventRecord(start_gpu);
    addKernel <<<block_size, thread_size>>> (dev_c, dev_a, dev_b, arraySize);
    cudaEventRecord(stop_gpu);


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Make sure the stop_gpu event is recorded before doing the time computation
    cudaEventSynchronize(stop_gpu);
    float gpu_runtime_ms;
    cudaEventElapsedTime(&gpu_runtime_ms, start_gpu, stop_gpu);

    if (checkCuda(cpu_result, c, arraySize)) {
        printf("GPU results are correct \n");
    }



    // 3. Compare execution time for the GPU and the CPU

    std::cout << "CPU time :" << cpu_runtime_us << " us" << std::endl;
    std::cout << "GPU time : " << gpu_runtime_ms * 1000 << " us" << std::endl;

    //TODO 3) Compute the speedup (cpu time / gpu time), 
    // compare the speedup for different vector size
    //float speedup = ...
    //std::cout << "speedup : " << speedup <<" %"<<std::endl;

    //3 vectors (2 inputs & 1 output) are stocked in the memory
    float memoryUsed = 3.0 * arraySize * sizeof(int);
    float memoryThroughput = memoryUsed / gpu_runtime_ms / 1e+6; //Divide by 1 000 000 to have GB/s

    //1 operation (+) is done for each element of the vectors
    float numOperation = 1.0 * arraySize;
    float computationThroughput = numOperation / gpu_runtime_ms / 1e+6; //Divide by 1 000 000 to have GOPS/s

    std::cout << "Memory throughput : " << memoryThroughput << " GB/s " << std::endl;
    std::cout << "Computation throughput : " << computationThroughput << " GOPS/s " << std::endl;

    free(a);
    free(b);
    free(c);
    free(cpu_result);

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

bool checkCuda(int* out_cpu, int* out_gpu, int N) {
    bool res = true;
    for (int i = 0; i < N; i++) {
        if (out_cpu[i] != out_gpu[i]) {
            printf("ERROR : cpu : %d != gpu : %d \n", out_cpu[i], out_gpu[i]);
            res = false;
        }
    }
    return res;
}

