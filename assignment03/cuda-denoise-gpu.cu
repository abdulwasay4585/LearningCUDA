/*
![Figure 1: Denoising example (original image by Simpsons, CC BY-SA 3.0, <https://commons.wikimedia.org/w/index.php?curid=8904364>).](denoise.png)

The file [cuda-denoise.c](cuda-denoise.c) contains a serial
implementation of an _image denoising_ algorithm that (to some extent)
can be used to "cleanup" color images. The algorithm replaces the
color of each pixel with the _median_ of the four adjacent pixels plus
itself (_median-of-five_).  The median-of-five algorithm is applied
separately for each color channel (red, green, and blue).

This is particularly useful for removing "hot pixels", i.e., pixels
whose color is way off its intended value, for example due to problems
in the sensor used to acquire the image. However, depending on the
amount of noise, a single pass could be insufficient to remove every
hot pixel; see Figure 1.

The goal of this exercise is to parallelize the denoising algorithm on
the GPU using CUDA. You should launch as many CUDA threads as pixels
in the image, so that each thread is mapped onto a different pixel.

The input image is read from standard input in
[PPM](http://netpbm.sourceforge.net/doc/ppm.html) (Portable Pixmap)
format; the result is written to standard output in the same format.

To compile:

        nvcc cuda-denoise.cu -o cuda-denoise

To execute:

        ./cuda-denoise < input > output

Example:

        ./cuda-denoise < valve-noise.ppm > valve-denoised.ppm

## Files

- [cuda-denoise.cu](cuda-denoise.cu) [hpc.h](hpc.h)
- [valve-noise.ppm](valve-noise.ppm) (sample input)

 ***/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "hpc.h"
typedef struct {
    int width;   /* Width of the image (in pixels) */
    int height;  /* Height of the image (in pixels) */
    int maxcol;  /* Largest color value (Used by the PPM read/write routines) */
    unsigned char *r, *g, *b; /* color channels (arrays of width x height elements each); each value must be less than or equal to maxcol */
} PPM_image;

/**
 * Read a PPM file from file `f`. This function is not very robust; it
 * may fail on perfectly legal PGM images, but works for the provided
 * cat.pgm file.
 */
void read_ppm( FILE *f, PPM_image* img )
{
    char buf[1024];
    const size_t BUFSIZE = sizeof(buf);
    char *s;
    int nread;

    assert(f != NULL);
    assert(img != NULL);

    /* Get the file type (must be "P6") */
    s = fgets(buf, BUFSIZE, f);
    if (0 != strcmp(s, "P6\n")) {
        fprintf(stderr, "FATAL: wrong file type %s\n", buf);
        exit(EXIT_FAILURE);
    }
    /* Get any comment and ignore it; does not work if there are
       leading spaces in the comment line */
    do {
        s = fgets(buf, BUFSIZE, f);
    } while (s[0] == '#');
    /* Get width, height */
    sscanf(s, "%d %d", &(img->width), &(img->height));
    /* get maxcol; must be less than or equal to 255 */
    s = fgets(buf, BUFSIZE, f);
    sscanf(s, "%d", &(img->maxcol));
    if ( img->maxcol > 255 ) {
        fprintf(stderr, "FATAL: maxcol=%d > 255\n", img->maxcol);
        exit(EXIT_FAILURE);
    }
    /* Get the binary data */
    img->r = (unsigned char*)malloc((img->width)*(img->height));
    assert(img->r != NULL);
    img->g = (unsigned char*)malloc((img->width)*(img->height));
    assert(img->g != NULL);
    img->b = (unsigned char*)malloc((img->width)*(img->height));
    assert(img->b != NULL);
    for (int k=0; k<(img->width)*(img->height); k++) {
        nread = fscanf(f, "%c%c%c", img->r + k, img->g + k, img->b + k);
        if (nread != 3) {
            fprintf(stderr, "FATAL: error reading pixel data\n");
            exit(EXIT_FAILURE);
        }
    }
}

/**
 * Write the image `img` to file `f`; is not NULL, use the string
 * `comment` as metadata.
 */
void write_ppm( FILE *f, const PPM_image* img, const char *comment )
{
    assert(f != NULL);
    assert(img != NULL);

    fprintf(f, "P6\n");
    fprintf(f, "# %s\n", comment != NULL ? comment : "");
    fprintf(f, "%d %d\n", img->width, img->height);
    fprintf(f, "%d\n", img->maxcol);
    for (int k=0; k<(img->width)*(img->height); k++) {
        fprintf(f, "%c%c%c", img->r[k], img->g[k], img->b[k]);
    }
}

/**
 * Free all memory used by the structure `img`
 */
void free_ppm( PPM_image* img )
{
    assert(img != NULL);
    free(img->r);
    free(img->g);
    free(img->b);
    img->r = img->g = img->b = NULL; /* not necessary */
    img->width = img->height = img->maxcol = -1;
}

#define BLKDIM 32

/**
 * Swap *a and *b if necessary so that, at the end, *a <= *b
 */

__device__ void compare_and_swap_dev( unsigned char *a, unsigned char *b )
{
    if (*a > *b ) {
        unsigned char tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

#define BLOCK_SIZE 16
#define SHARED_SIZE (BLOCK_SIZE + 2)

__global__ void denoise_kernel(const unsigned char *in, unsigned char *out, int width, int height) {
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    __shared__ unsigned char smem[SHARED_SIZE][SHARED_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Load center
    if (x < width && y < height) {
        smem[ty + 1][tx + 1] = in[y * width + x];
    } else {
        smem[ty + 1][tx + 1] = 0;
    }

    // Load halo for left/right
    if (tx == 0) {
        smem[ty + 1][0] = (x > 0 && y < height) ? in[y * width + (x - 1)] : 0;
    } else if (tx == BLOCK_SIZE - 1) {
        smem[ty + 1][SHARED_SIZE - 1] = (x < width - 1 && y < height) ? in[y * width + (x + 1)] : 0;
    }

    // Load halo for top/bottom
    if (ty == 0) {
        smem[0][tx + 1] = (y > 0 && x < width) ? in[(y - 1) * width + x] : 0;
    } else if (ty == BLOCK_SIZE - 1) {
        smem[SHARED_SIZE - 1][tx + 1] = (y < height - 1 && x < width) ? in[(y + 1) * width + x] : 0;
    }

    __syncthreads();

    // Compute median
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        unsigned char v[5];
        v[0] = smem[ty + 1][tx + 1];
        v[1] = smem[ty + 1][tx];
        v[2] = smem[ty + 1][tx + 2];
        v[3] = smem[ty][tx + 1];
        v[4] = smem[ty + 2][tx + 1];

        // Partial sort to find median (bubble sort)
        compare_and_swap_dev( v+3, v+4 );
        compare_and_swap_dev( v+2, v+3 );
        compare_and_swap_dev( v+1, v+2 );
        compare_and_swap_dev( v  , v+1 );
        compare_and_swap_dev( v+3, v+4 );
        compare_and_swap_dev( v+2, v+3 );
        compare_and_swap_dev( v+1, v+2 );
        compare_and_swap_dev( v+3, v+4 );
        compare_and_swap_dev( v+2, v+3 );

        out[y * width + x] = v[2];
    } else if (x < width && y < height) {
        // Border pixels remain unchanged
        out[y * width + x] = smem[ty + 1][tx + 1];
    }
}

void denoise_gpu(unsigned char *r, unsigned char *g, unsigned char *b, int width, int height) {
    int img_size = width * height * sizeof(unsigned char);
    unsigned char *d_r_in, *d_r_out, *d_g_in, *d_g_out, *d_b_in, *d_b_out;

    cudaSafeCall(cudaMalloc((void**)&d_r_in, img_size));
    cudaSafeCall(cudaMalloc((void**)&d_r_out, img_size));
    cudaSafeCall(cudaMalloc((void**)&d_g_in, img_size));
    cudaSafeCall(cudaMalloc((void**)&d_g_out, img_size));
    cudaSafeCall(cudaMalloc((void**)&d_b_in, img_size));
    cudaSafeCall(cudaMalloc((void**)&d_b_out, img_size));

    cudaSafeCall(cudaMemcpy(d_r_in, r, img_size, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_g_in, g, img_size, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_b_in, b, img_size, cudaMemcpyHostToDevice));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    denoise_kernel<<<grid, threads>>>(d_r_in, d_r_out, width, height);
    cudaCheckError();
    denoise_kernel<<<grid, threads>>>(d_g_in, d_g_out, width, height);
    cudaCheckError();
    denoise_kernel<<<grid, threads>>>(d_b_in, d_b_out, width, height);
    cudaCheckError();

    cudaSafeCall(cudaMemcpy(r, d_r_out, img_size, cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(g, d_g_out, img_size, cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(b, d_b_out, img_size, cudaMemcpyDeviceToHost));

    cudaFree(d_r_in); cudaFree(d_r_out);
    cudaFree(d_g_in); cudaFree(d_g_out);
    cudaFree(d_b_in); cudaFree(d_b_out);
}

int main( void )
{
    PPM_image img;
    read_ppm(stdin, &img);
    const double tstart = hpc_gettime();
    
    denoise_gpu(img.r, img.g, img.b, img.width, img.height);
    cudaDeviceSynchronize();
    
    const double elapsed = hpc_gettime() - tstart;
    fprintf(stderr, "CPU + GPU Execution time %.3f\n", elapsed);
    write_ppm(stdout, &img, "produced by cuda-denoise-gpu.cu");
    free_ppm(&img);
    return EXIT_SUCCESS;
}
