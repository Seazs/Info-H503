
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vector_functions.h"

#include <iostream>
#include <vector>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// CUDA alignment is set to 16 to support the largest load type that we have =>
// uint4 (4*4 bytes) => 16 bit
#define CUDA_ALIGNMENT 16

#define CHK(code) \
do { \
    if ((code) != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s %s %i\n", \
                        cudaGetErrorString((code)), __FILE__, __LINE__); \
        goto Error; \
    } \
} while (0)

//return ceiling(x/y)
inline unsigned divUp(unsigned x, unsigned y) { return (x + y - 1) / y; }


__constant__ __device__ float rgb_coeff[4] = { 0.2126f, 0.7152f, 0.0722f, 0.0f };



__global__ void rgba2grayscale(const uint8_t *__restrict__ input, uint8_t *__restrict__ output, unsigned h, unsigned w, unsigned in_pitch, unsigned out_pitch)
{
    const unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    constexpr int channel = 4;
    if (idx_x < w && idx_y < h) {
        unsigned idx = idx_y * in_pitch + idx_x * channel;
        float gray = rgb_coeff[0] * input[idx] +rgb_coeff[1] * input[idx + 1] +
            rgb_coeff[2] * input[idx + 2] + rgb_coeff[3] * input[idx + 3];
        output[idx_y * out_pitch + idx_x] = static_cast<char>(gray);
    }
}

__global__ void rgba2grayscale_multiple_values(const unsigned char * __restrict__ input, unsigned char * __restrict__ output, unsigned h, unsigned w, unsigned in_pitch, unsigned out_pitch, int values_per_thread)
{
    
    const unsigned idx_x = (blockIdx.x * blockDim.x + threadIdx.x)*values_per_thread;
    const unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    constexpr unsigned channel = 4;

    unsigned idx = idx_y * in_pitch + idx_x * channel;
    unsigned idx_out = idx_y * out_pitch + idx_x;

    if (idx_y < h) {
        for (unsigned i = 0; i < values_per_thread; idx += channel, ++i) {
            if ((idx_x + i) < out_pitch) {
                float gray = rgb_coeff[0] * input[idx] +
                    rgb_coeff[1] * input[idx + 1] +
                    rgb_coeff[2] * input[idx + 2] +
                    rgb_coeff[3] * input[idx + 3];
                output[idx_out + i] = static_cast<char>(gray);
            }
        }
    }
}



__global__ void rgba2grayscale_wide(const uint4 *__restrict__ input, uchar4 * __restrict__ output, unsigned h, unsigned w, unsigned in_pitch, unsigned out_pitch, int values_per_thread)
{
    //Number of pixels processed in a thread must be a multiple of 4
    unsigned total_memory_ops = values_per_thread / 4;

    const unsigned idx_x = (blockIdx.x * blockDim.x + threadIdx.x) * total_memory_ops;
    const unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    constexpr unsigned channel = 4;

    //Process 4 pixels with a single load : 16 B = 4*rgba = 4*4*1 Bytes
    constexpr unsigned values_per_load = sizeof(uint4); //16 B
    //Write the results for 4 pixels with a single store : 4*1 uchar = 4 B
    constexpr unsigned values_per_store = sizeof(uchar4); //4 B

    //1 uint4 = 16 char/uint8 = 16 B : for the same size in memory, array of uint4 has less elements than array of char
    //-> pitch of array changes with the type
    const unsigned stride_in = in_pitch / values_per_load;
    const unsigned stride_out = out_pitch / values_per_store;


    if (idx_y < h) {
        for (int mem_op = 0; mem_op < total_memory_ops; ++mem_op) {
            if((idx_x + mem_op) < stride_in){
                union {
                    uint4 rgba_raw;
                    uchar4 rgba[4];
                };
                union {
                    uchar4 gray_out;
                    unsigned char gray_split_pixel[4];
                };
                //load 4 pixels
                rgba_raw = input[idx_y * stride_in + idx_x + mem_op];
                for (int i = 0; i < 4; ++i) {
                    //process each pixel, use uchar4 type to access rgba values
                    if ((idx_x + mem_op) * 4 + i < w) {
                        float gray = rgb_coeff[0] * rgba[i].x +
                            rgb_coeff[1] * rgba[i].y +
                            rgb_coeff[2] * rgba[i].z +
                            rgb_coeff[3] * rgba[i].w;
                        //use char [4] to store result for each pixel
                        gray_split_pixel[i] = (char)gray;
                    }                    
                }

               //use uchar4 type to write the results of 4 pixel with a single store operation
                const int idx_out = idx_y * stride_out + idx_x + mem_op;
                output[idx_out] = gray_out;
            }
        }
    }

}

bool checkResult(unsigned char* ref, unsigned char* img, int w, int h) {
    bool res = 1;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            unsigned char r = ref[y * w + x];
            unsigned char i = img[y * w + x];
            if (abs(r -i) > 1) { //tolerance of 1 because rounding between GPU and CPU might give different results
                res = 0;
            }   
        }
    }
    return res;
}


int main()
{
    const char* filename = "../images/input_4K.png";
    const char* filename_out = "../images/output.png";
    //load png
    int h, w, channel;
    unsigned char* img = stbi_load(filename, &w, &h, &channel, 4);

    if (img == NULL) {
        printf("Error loading img \n");
        const char* reason = stbi_failure_reason();
        printf(reason);
    }

    unsigned char* out_img = (unsigned char*)malloc(w*h*sizeof(unsigned char));
    unsigned char* ref_img = (unsigned char*)malloc(w*h*sizeof(unsigned char));

    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            int pix = y * w * channel + x * channel;
            float gray = img[pix] * 0.2126f + img[pix + 1] * 0.7152f +
                img[pix + 2] * 0.0722f + img[pix + 3] * 0.0f;
            ref_img[y * w + x] = static_cast<uint8_t>(gray);
        }
    }


    //Make sure the data is aligned correctly to use vectorized load/store
    unsigned pitch_device_rgb = divUp(w * channel, CUDA_ALIGNMENT) * CUDA_ALIGNMENT;
    unsigned pitch_device_gray = divUp(w, CUDA_ALIGNMENT) * CUDA_ALIGNMENT;

    uint8_t *d_input, *d_output;

    CHK(cudaSetDevice(0));

    CHK(cudaMalloc(&d_input, h * pitch_device_rgb)); // w*h*4
    CHK(cudaMalloc(&d_output, h * pitch_device_gray)); //w*h*1
    
 
    CHK(cudaMemcpy2D(d_input, pitch_device_rgb, img, w * channel, w * channel, h, cudaMemcpyHostToDevice));
    
    
    dim3 thread_per_block_c(1, 256,1);
    dim3 block_per_grid_c(divUp(w, thread_per_block_c.x), divUp(h, thread_per_block_c.y));

    rgba2grayscale << <block_per_grid_c, thread_per_block_c >> > (d_input, d_output, h, w, pitch_device_rgb, pitch_device_gray);

    CHK(cudaGetLastError());
    CHK(cudaMemcpy2D(out_img, w, d_output, pitch_device_gray, w, h, cudaMemcpyDeviceToHost));

    if (!checkResult(ref_img, out_img, w, h)) {
        printf("Vertical blocks : Wrong GPU results \n");
    }
    

    /* --- ROWS ---*/
    dim3 thread_per_block_r(256, 1, 1);
    dim3 block_per_grid_r(divUp(w, thread_per_block_r.x), divUp(h, thread_per_block_r.y));

    //reset the output image to black
    CHK(cudaMemset(d_output, 0, h * pitch_device_gray * sizeof(unsigned char)));

    rgba2grayscale << <block_per_grid_r, thread_per_block_r >> > (d_input, d_output, h, w, pitch_device_rgb, pitch_device_gray);

    CHK(cudaGetLastError());
    CHK(cudaMemcpy2D(out_img, w, d_output, pitch_device_gray, w, h, cudaMemcpyDeviceToHost));

    if (!checkResult(ref_img, out_img, w, h)) {
        printf("Horizontal blocks : Wrong GPU results \n");
    }
    

    /* Several element per thread */
    int value_per_thread = 4;

    dim3 thread_per_block_m(32, 4, 1);
    dim3 block_per_grid_m(divUp(w, thread_per_block_m.x * value_per_thread), divUp(h, thread_per_block_m.y));

    //reset the output image to black
    CHK(cudaMemset(d_output, 0, h * pitch_device_gray * sizeof(unsigned char)));

    rgba2grayscale_multiple_values << <block_per_grid_m, thread_per_block_m >> > (d_input, d_output, h, w, pitch_device_rgb, pitch_device_gray, value_per_thread);
    
    CHK(cudaGetLastError());
    CHK(cudaMemcpy2D(out_img, w, d_output, pitch_device_gray, w, h, cudaMemcpyDeviceToHost));

    if (!checkResult(ref_img, out_img, w, h)) {
        printf("Multiple pixel per thread : Wrong GPU results \n");
    }

    

    /* Vectorized operation */

    //Number of pixels processed in a thread must be a multiple of 4
    value_per_thread = 4;

    dim3 thread_per_block_w(32, 4, 1);
    dim3 block_per_grid_w(divUp(w, thread_per_block_w.x * value_per_thread), divUp(h, thread_per_block_w.y));

    //reset the output image to black
    CHK(cudaMemset(d_output, 0, h * pitch_device_gray * sizeof(unsigned char)));

    rgba2grayscale_wide << <block_per_grid_w, thread_per_block_w >> > (reinterpret_cast<uint4*>(d_input), reinterpret_cast<uchar4*>(d_output),
        h, w, pitch_device_rgb, pitch_device_gray, value_per_thread);
    CHK(cudaGetLastError());
        
    CHK(cudaMemcpy2D(out_img, w, d_output, pitch_device_gray, w, h, cudaMemcpyDeviceToHost));

    if (!checkResult(ref_img, out_img, w, h)) {
        printf("Vectorized operation : Wrong GPU results \n");
    }

    stbi_write_png(filename_out, w, h, 1, out_img, w);

   
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    CHK(cudaDeviceReset());
    
    

Error:
    free(out_img);
    free(ref_img);
    return 0;
}


