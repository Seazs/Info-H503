
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vector_functions.h"


#include <iostream>
#include <direct.h>
#include <vector>
#include <stdio.h>

//TODO : make sure you add the header file to visual studio project, see appendice slide
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// CUDA alignment is set to 16 to support the largest load type that we have =>
// uint4 (4*4 bytes) => 16 bit
#define CUDA_ALIGNMENT 16


using namespace std;


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



__global__ void rgba2grayscale(const uint8_t* __restrict__ input, uint8_t* __restrict__ output, unsigned h, unsigned w, unsigned in_pitch, unsigned out_pitch)
{
    /* TODO : implement naive kernel
    1 : compute the indices in x & y
    2 : compute the global indices in the input and output image
    3 : process the pixel */

    const unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_x < w && idx_y < h) {
        constexpr int channel = 4; // le nombre de channel différent (rgb et alpha) 
        unsigned idx = idx_y * in_pitch + idx_x * channel; // l'image est stocké ligne par ligne et pixel par pixel. en mettant toutes les valeurs du pixel (rgbalpha) les une a la suite des autres.
        float gray = rgb_coeff[0] * input[idx] + rgb_coeff[1] * input[idx + 1] + rgb_coeff[2] * input[idx + 2] + rgb_coeff[3] * input[idx + 3];

        output[idx_y * out_pitch + idx_x] = static_cast<char>(gray);
    }
}

__global__ void rgba2grayscale_multiple_values(const unsigned char* __restrict__ input, unsigned char* __restrict__ output, unsigned h, unsigned w, unsigned in_pitch, unsigned out_pitch, int values_per_thread)
{

    /* TODO : implement a kernel where 1 thread process several pixels
    1 : compute the indices in x & y, take into account that a thread process several pixels
    2 : compute the global indices in the input and output image
    3 : process the pixelS
    */

    const unsigned idx_x = (blockIdx.x * blockDim.x + threadIdx.x) * values_per_thread;
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



__global__ void rgba2grayscale_wide(const uint4* __restrict__ input, uchar4* __restrict__ output, unsigned h, unsigned w, unsigned in_pitch, unsigned out_pitch, int values_per_thread)
{
    /* TODO : implement a kernel where 1 thread process several pixels, using vectorized load & store
  1 : compute the indices in x & y, take into account that a thread process several pixels
  2 : compute the global indices in the input and output image, be carefull for the same size in memory, an array of uint4 has less elements than an array of char
  3 : process the pixelS, use the keyword union to access the same data using different types
  */



}

bool checkResult(unsigned char* ref, unsigned char* img, int w, int h) {
    bool res = 1;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            unsigned char r = ref[y * w + x];
            unsigned char i = img[y * w + x];
            if (abs(r - i) > 1) { //tolerance of 1 because rounding between GPU and CPU might give different results
                res = 0;
            }
        }
    }
    return res;
}


int main()
{
    const char* filename = "./input_4K.png";
    const char* filename_out = "./output.png";
    // Define a buffer 
    const size_t size = 1024;
    // Allocate a character array to store the directory path
    char buffer[size];

    // Call _getcwd to get the current working directory and store it in buffer
    if (getcwd(buffer, size) != NULL) {
        // print the current working directory
        cout << "Current working directory: " << buffer << endl;
    }

    //load png
    int h, w, channel;
    unsigned char* img = stbi_load(filename, &w, &h, &channel, 4); //Make sure you import the stb_image librairy

    if (img == NULL) {
        printf("Error loading img \n");
        const char* reason = stbi_failure_reason();
        printf(reason);
    }

    unsigned char* out_img = (unsigned char*)malloc(w * h * sizeof(unsigned char));
    unsigned char* ref_img = (unsigned char*)malloc(w * h * sizeof(unsigned char));

    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            int pix = y * w * channel + x * channel;
            float gray = img[pix] * 0.2126f + img[pix + 1] * 0.7152f + img[pix + 2] * 0.0722f + img[pix + 3] * 0.0f;
            ref_img[y * w + x] = static_cast<char>(gray);
        }
    }


    //Define the width of the 2D array used to put the image in memory
    //Make sure the data is aligned correctly to use vectorized load/store
    unsigned pitch_device_rgb = divUp(w * channel, CUDA_ALIGNMENT) * CUDA_ALIGNMENT;
    unsigned pitch_device_gray = divUp(w, CUDA_ALIGNMENT) * CUDA_ALIGNMENT;

    uint8_t* d_input, * d_output;

    CHK(cudaSetDevice(0));

    CHK(cudaMalloc(&d_input, h * pitch_device_rgb));
    CHK(cudaMalloc(&d_output, h * pitch_device_gray));

    //Copy the input image in memory
    CHK(cudaMemcpy2D(d_input, pitch_device_rgb, img, w * channel, w * channel, h, cudaMemcpyHostToDevice));


    ///* --- VERTICAL BLOCKS --- */
    //dim3 thread_per_block_c(1,256,1);
    //dim3 block_per_grid_c(divUp(w, thread_per_block_c.x), divUp(h, thread_per_block_c.y));

    //rgba2grayscale << <block_per_grid_c, thread_per_block_c >> > (d_input, d_output, h, w, pitch_device_rgb, pitch_device_gray);

    //CHK(cudaGetLastError());
    ////copy the output image on the CPU
    //CHK(cudaMemcpy2D(out_img, w, d_output, pitch_device_gray, w, h, cudaMemcpyDeviceToHost));

    //if (!checkResult(ref_img, out_img, w, h)) {
    //    printf("Vertical blocks : Wrong GPU results \n");
    //}else{printf("Vertical blocks : right GPU results \n");
    //}


    ///* --- HORIZONTAL BLOCKS ---*/
    //dim3 thread_per_block_r(256, 1, 1);
    //dim3 block_per_grid_r(divUp(w, thread_per_block_r.x), divUp(h, thread_per_block_r.y));

    ////reset the output image to black
    //CHK(cudaMemset(d_output, 0, h * pitch_device_gray * sizeof(unsigned char)));

    //rgba2grayscale << <block_per_grid_r, thread_per_block_r >> > (d_input, d_output, h, w, pitch_device_rgb, pitch_device_gray);

    //CHK(cudaGetLastError());
    //CHK(cudaMemcpy2D(out_img, w, d_output, pitch_device_gray, w, h, cudaMemcpyDeviceToHost));

    //if (!checkResult(ref_img, out_img, w, h)) {
    //    printf("Horizontal blocks : Wrong GPU results \n");
    //}
    //else {
    //    printf("Vertical blocks : right GPU results \n");
    //}


    /* --- PROCESS MULTIPLE PIXELS PER THREAD --- */
    int value_per_thread = 4;

    dim3 thread_per_block_m(32, 4, 1);
    dim3 block_per_grid_m(divUp(w, thread_per_block_m.x * value_per_thread), divUp(h, thread_per_block_m.y)); // ca permet de savoir combien on veut de block. on donne la dimension de la grille. Etant donnée qu'on traite 4 vauleurs par thread et qu'on a 32 * 4 = 128 
                                                                                                              //threads par block, il suffit de diviser la largeur et la hauteur par le nombre de threads qu'on a pas ligne et colonne respectivement.

    //reset the output image to black
    CHK(cudaMemset(d_output, 0, h * pitch_device_gray * sizeof(unsigned char)));

    //call the kernel, check for errors and copy the reuslt on the CPU
    rgba2grayscale_multiple_values << <block_per_grid_m, thread_per_block_m >> > (d_input, d_output, h, w, pitch_device_rgb, pitch_device_gray, value_per_thread);

    CHK(cudaGetLastError());
    CHK(cudaMemcpy2D(out_img, w, d_output, pitch_device_gray, w, h, cudaMemcpyDeviceToHost));



    if (!checkResult(ref_img, out_img, w, h)) {
        printf("Multiple pixel per thread : Wrong GPU results \n");
    }
    else {
            printf("Vertical blocks : right GPU results \n");
        }



    ///* --- VECTORIZED OPERATIONS --- */

    ////Number of pixels processed in a thread must be a multiple of 4
    //value_per_thread; // ... 

    //dim3 thread_per_block_w(/*TODO*/);
    //dim3 block_per_grid_w(/*TODO*/);

    ////reset the output image to black
    //CHK(cudaMemset(/*TODO*/));

    ////call the kernel, don't forget to cast the input and output to the right type of pointer
    ////check for errors and copy the reuslt on the CPU


    //if (!checkResult(ref_img, out_img, w, h)) {
    //    printf("Vectorized operation : Wrong GPU results \n");
    //}

    //write the output image, make sure you include the librairy stb_image_write
    stbi_write_png(filename_out, w, h, 1, out_img, w);


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    CHK(cudaDeviceReset());



Error:
    free(out_img);
    free(ref_img);
    return 0;
}


