
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <atomic>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define VERBOSE true

#define CHK(code) \
do { \
    if ((code) != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s %s %i\n", \
                        cudaGetErrorString((code)), __FILE__, __LINE__); \
        goto Error; \
    } \
} while (0)



void histogramCPU(unsigned char* img, unsigned int* hist, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        int value = img[i];
        hist[value] += 1;
    }
}

__global__ void histogramNaive(unsigned char* img, unsigned int* hist, long N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx > N)
        return;

    unsigned int color = img[idx];

    atomicAdd(&(hist[color]), 1);

}

__global__ void histogramStride(unsigned char* img, unsigned int* hist, long N) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int width = blockDim.x * gridDim.x;

    if (idx >= N)
        return;

    while (idx < N) {
        unsigned int val = img[idx];
        
        atomicAdd(&hist[val], 1);

        idx += width;

    }
}

__global__ void histogramPerBlock(unsigned char* img, unsigned int* hist, long N) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int width = blockDim.x * gridDim.x;

    unsigned int* bloc_hist = hist + blockIdx.x * 256;

    if (idx >= N)
        return;

    while (idx < N) {
        unsigned int val = img[idx];

        atomicAdd(&bloc_hist[val], 1);

        idx += width;

    }
    if (blockIdx.x > 0) {
        __syncthreads();
        atomicAdd(&(hist[threadIdx.x]), bloc_hist[threadIdx.x]);
    }






}
__global__ void histogramSharedMem(unsigned char* img, unsigned int* hist, long N) {
    //Use the shared memory to compute one histogram per block
    __shared__ unsigned int* local_hist[256];
    local_hist[threadIdx.x] = 0; // each thread will initialise 1 value

    __syncthreads(); // faut faire la synchro avant de commencer, sinon on pourrait avoir des threads qui viennent réinitiliser une valeur à 0 alors qu'elle avait déjà été modifié par un autre thread

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int width = blockDim.x * gridDim.x;

    while (idx < N) {
        unsigned int val = img[idx];

        atomicAdd(&local_hist[val], 1);

        idx += width;
    }
    __syncthreads();
    // each threads will be charged to copy one value on the histogram to the global memory
    atomicAdd(&hist[threadIdx.x], local_hist[threadIdx.x]);



    //TODO
}

void saveHist(unsigned int* hist, char* name) {
    std::ofstream fout;
    fout.open(name);
    if (!fout.good()) {
        printf("Error opening .csv");
    }
    fout << hist[0];
    for (int i = 1; i < 256; i++) {
        fout << "," << hist[i];
    }
    fout.close();
}

bool checkComputation(const unsigned int* cpu_res, const unsigned int* gpu_res, size_t N) {
    bool res = true;
    for (int i = 0; i < N; i++) {
        if (cpu_res[i] != gpu_res[i]) {
            res = false;
            if (VERBOSE)
                printf("ERROR idx %d, CPU : %d vs GPU : %d \n", i, cpu_res[i], gpu_res[i]);
        }
    }
    return res;
}

int main()
{
    //Load image
    int width, height, ch;
    unsigned char* img = stbi_load("Lena.png", &width, &height, &ch, 1);

    if (img == NULL) {
        printf("Error loading img \n");
        const char* reason = stbi_failure_reason();
        printf(reason);
    }

    //Compute result with c++
    unsigned int histogramC[256] = { 0 };

    float cpu_runtime_us = 0;
    std::chrono::steady_clock::time_point start_cpu = std::chrono::high_resolution_clock::now();
    histogramCPU(img, histogramC, width, height);
    std::chrono::steady_clock::time_point stop_cpu = std::chrono::high_resolution_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu).count();
    cpu_runtime_us = (float)runtime;

    saveHist(histogramC, "hist_c++.csv");


    int size = width * height;
    unsigned int histCuda[256] = { 0 };

    unsigned char* d_img = 0;
    unsigned int* d_hist = 0;


    CHK(cudaSetDevice(0));

    //Init memory
    CHK(cudaMalloc((void**)&d_img, size * sizeof(unsigned char)));
    CHK(cudaMalloc((void**)&d_hist, 256 * sizeof(unsigned int)));

    //Copy image to GPU
    CHK(cudaMemcpy(d_img, img, size * sizeof(unsigned char), cudaMemcpyHostToDevice));

    //cuda event used to time the kernel
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    //define blocks of 256 thread
    //more convenient when working with the shared memory
    int N_threads = 256;


    /* Naive version */
    //Number of blocks needed for naive version
    int N_blocks = (size + (N_threads - 1)) / N_threads;

    dim3 block_size(N_blocks);
    dim3 thread_size(N_threads);

    float naive_kernel_runtime_ms = 0;

    cudaEventRecord(start_gpu);
    histogramNaive << <block_size, thread_size >> > (d_img, d_hist, (long)size);
    cudaEventRecord(stop_gpu);

    CHK(cudaGetLastError());
    CHK(cudaDeviceSynchronize());
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&naive_kernel_runtime_ms, start_gpu, stop_gpu);

    CHK(cudaMemcpy(histCuda, d_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    //Check if GPU acceleration still gives the correct results
    if (!checkComputation(histogramC, histCuda, 256)) {
        printf("GPU naive results incorrect\n");
    }

    saveHist(histCuda, "hist_GPUnaive.csv");


    /* Stride version*/
    //Number of blocks needed for stride version
    int N_blocksCol = (width * 16 + (N_threads - 1)) / N_threads;   // le width * 4, c'est parce que j'ai décidé qu'un threads allait traiter 1/4 de colonne. Donc on a besoin d'un nombre de bloc = 4* nombre de colonne
    dim3 block_sizeStride(N_blocksCol);
    dim3 thread_sizeStride(N_threads);


    //Reset mem device to 0 before launching the kernel (kernel use add)
    CHK(cudaMemset(d_hist, 0, 256 * sizeof(unsigned int)));

    float stride_kernel_runtime_ms = 0;

    cudaEventRecord(start_gpu);
    histogramStride <<  <block_sizeStride, thread_sizeStride >> > (d_img, d_hist, (long)size);
    cudaEventRecord(stop_gpu);

    CHK(cudaGetLastError());
    CHK(cudaDeviceSynchronize());
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&stride_kernel_runtime_ms, start_gpu, stop_gpu);


    CHK(cudaMemcpy(histCuda, d_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    //Check if GPU acceleration still gives the correct results
    if (!checkComputation(histogramC, histCuda, 256)) {
        printf("GPU naive results incorrect\n");
    }

    saveHist(histCuda, "hist_GPUStride.csv");




    //TODO

    /*Stride Global*/
    //Number of blocks needed for stride version
    N_blocksCol = (width * 16 + (N_threads - 1)) / N_threads;
    dim3 block_sizeOwnHist(N_blocksCol);


    //Need enough memory space to stock the histogram computed by each block
    unsigned int* d_hist_long;
    CHK(cudaMalloc((void**)&d_hist_long, N_blocksCol*256* sizeof(unsigned int))); // allouer n histograme de mémoire, ou n est le nombre de block étant donnée que chaque bloc va calculer son propre histograme et puis tout sommer a la fin
    CHK(cudaMemset(d_hist, 0, 256 * sizeof(unsigned int)));


    float global_kernel_runtime_ms = 0;
    cudaEventRecord(start_gpu);
    histogramPerBlock << <block_sizeOwnHist, thread_size >> > (d_img, d_hist_long, (long)size);
    cudaEventRecord(stop_gpu);


    CHK(cudaGetLastError());
    CHK(cudaDeviceSynchronize());
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&global_kernel_runtime_ms, start_gpu, stop_gpu);

    CHK(cudaMemcpy(histCuda, d_hist_long, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    //Check if GPU acceleration still gives the correct results
    if (!checkComputation(histogramC, histCuda, 256)) {
        printf("GPU global results incorrect\n");
    }
    saveHist(histCuda, "hist_GPUglobal.csv");




    ////TODO

    /*Shared memory*/

    //Number of blocks needed for stride version
    N_blocksCol = (width * 16 + (N_threads - 1)) / N_threads;
    dim3 block_sizeShared(N_blocksCol);

    cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));



    float shared_kernel_runtime_ms = 0;


    cudaEventRecord(start_gpu);
    histogramSharedMem << <block_sizeShared, thread_size >> > (d_img, d_hist, (long)size);
    cudaEventRecord(stop_gpu);

    CHK(cudaGetLastError());
    CHK(cudaDeviceSynchronize());
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&shared_kernel_runtime_ms, start_gpu, stop_gpu);

    CHK(cudaMemcpy(histCuda, d_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    //Check if GPU acceleration still gives the correct results
    if (!checkComputation(histogramC, histCuda, 256)) {
        printf("GPU shared mem results incorrect\n");
    }
    saveHist(histCuda, "hist_GPUshared.csv");


    ////TODO


    printf("CPU time %f us \n", cpu_runtime_us);
    printf("GPU - basic kernel time %f us, speedup : %f \n", naive_kernel_runtime_ms * 1000, cpu_runtime_us / (naive_kernel_runtime_ms * 1000));
    printf("GPU - stride kernel time %f us, speedup : %f \n", stride_kernel_runtime_ms * 1000, cpu_runtime_us / (stride_kernel_runtime_ms * 1000));
    printf("GPU - global kernel time %f us, speedup : %f \n", global_kernel_runtime_ms * 1000, cpu_runtime_us / (global_kernel_runtime_ms * 1000));
    printf("GPU - shared kernel time %f us, speedup : %f \n", shared_kernel_runtime_ms * 1000, cpu_runtime_us / (shared_kernel_runtime_ms * 1000));


Error:
    cudaFree(d_img);
    cudaFree(d_hist);
    cudaFree(d_hist_long);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
