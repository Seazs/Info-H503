
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <iostream>


bool checkCuda(int* out_cpu, int* out_gpu, int N);

#define CHK(code) \
do { \
    if ((code) != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s %s %i\n", \
                        cudaGetErrorString((code)), __FILE__, __LINE__); \
        goto Error; \
    } \
} while (0)

#define LOOP 1

__global__ void addKernel(int* c, const int* a, const int* b, int N, int j)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    //TODO mapping : modify to process more than 1 element per thread
    int index;
    for (int i = 0; i <j ; i++) {
        index = idx * j + i;
        if (index >= N)
            return;
        int va = a[index];
        int vb = b[index];
        //TODO compute intensity : use a loop to artificially increase the number of operation done
        int res = va + vb;
        c[index] = res;
    }
    

}

//TODO occupancy : test different number of thread/block
#define N_THREAD 128

int main()
{
    //TODO occupancy : test different vector size
    const int arraySize = 4096 * 4096;

    int ElemProcessedPerThread = 3;

    std::cout << "arraySize : " << arraySize << std::endl;
    int* h_a = new int[arraySize];  //Declare array dynamically to use bigger vector
    int* h_b = new int[arraySize];
    for (int i = 0; i < arraySize; i++) {
        h_a[i] = i;
        h_b[i] = arraySize - i;
    }
    int* h_c = new int[arraySize];
    int* h_cpu_result = new int[arraySize];

    //Computation on CPU
    std::chrono::steady_clock::time_point start_cpu = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < arraySize; i++) {
        h_cpu_result[i] = h_a[i] + h_b[i];
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
    dim3 block_size(((arraySize/ElemProcessedPerThread) + (N_THREAD - 1)) / N_THREAD);
    dim3 thread_size(N_THREAD);

    // Choose which GPU to run on, change this on a multi-GPU system.
    CHK(cudaSetDevice(0));

    // Allocate GPU buffers for three vectors (two input, one output)    .
    CHK(cudaMalloc((void**)&dev_c, arraySize * sizeof(int)));
    CHK(cudaMalloc((void**)&dev_a, arraySize * sizeof(int)));
    CHK(cudaMalloc((void**)&dev_b, arraySize * sizeof(int)));

    // Copy input vectors from host memory to GPU buffers.
    CHK(cudaMemcpy(dev_a, h_a, arraySize * sizeof(int), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dev_b, h_b, arraySize * sizeof(int), cudaMemcpyHostToDevice));;

    // Launch a kernel on the GPU with one thread for each element and time the kernel
    cudaEventRecord(start_gpu);
    addKernel << <block_size, thread_size >> > (dev_c, dev_a, dev_b, arraySize, ElemProcessedPerThread);
    cudaEventRecord(stop_gpu);


    // Check for any errors launching the kernel
    CHK(cudaGetLastError());


    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    CHK(cudaDeviceSynchronize());

    //TODO Benchmarking : run the kernel several time and take the average to obtain accurate and rpelicable results
    // Copy output vector from GPU buffer to host memory.
    CHK(cudaMemcpy(h_c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost));


    // Make sure the stop_gpu event is recorded before doing the time computation
    cudaEventSynchronize(stop_gpu);
    float gpu_runtime_ms;
    cudaEventElapsedTime(&gpu_runtime_ms, start_gpu, stop_gpu);

    if (checkCuda(h_cpu_result, h_c, arraySize)) {
        printf("GPU results are correct \n");
    }

    // 3. Compare execution time for the GPU and the CPU

    std::cout << "CPU time :" << cpu_runtime_us << " us" << std::endl;
    std::cout << "GPU time : " << gpu_runtime_ms * 1000 << " us" << std::endl;


    float speedup = cpu_runtime_us / (gpu_runtime_ms * 1000);
    std::cout << "speedup : " << speedup << std::endl;


    float memoryUsed = 3.0 * arraySize * sizeof(int);
    float memoryThroughput = memoryUsed / gpu_runtime_ms / 1e+6; //Divide by 1 000 000 to have GB/s

    float numOperation = 1.0 * arraySize * LOOP;
    float computationThroughput = numOperation / gpu_runtime_ms / 1e+6; //Divide by 1 000 000 to have GOPS/s

    std::cout << "Memory throughput : " << memoryThroughput << " GB/s " << std::endl;
    std::cout << "Computation throughput : " << computationThroughput << " GOPS/s " << std::endl;

    //TODO compute intensity : determine the compute intensity (hint : the unit are OPS/Byte)
    //float computeIntensity = ...
    //std::cout << "Compute intensity : " << computeIntensity << " OPS/Byte" << std::endl;


Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    delete[] h_a, h_b, h_c;

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
            printf("%d, ERROR : cpu : %d != gpu : %d \n", i, out_cpu[i], out_gpu[i]);
            res = false;
        }
    }
    return res;
}

