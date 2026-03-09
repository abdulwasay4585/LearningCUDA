/*
![Result of the Sobel operator](edge-detect.png)

The [Sobel operator](https://en.wikipedia.org/wiki/Sobel_operator) is
used to detect the edges on an grayscale image. The idea is to compute
the gradient of color change across each pixel; those pixels for which
the gradient exceeds a user-defined threshold are considered to be
part of an edge. Computation of the gradient involves the application
of a $3 \times 3$ stencil to the input image.

The program reads an input image fro standard input in
[PGM](https://en.wikipedia.org/wiki/Netpbm#PGM_example) (_Portable
Graymap_) format and produces a B/W image to standard output. The user
can specify an optional threshold on the command line.

The goal of this exercise is to parallelize the computation of the
Sobel operator using CUDA; this can be achieved by writing a kernel
that computes the edge at each pixel, and invoke the kernel from the
`edge_detect()` function.

To compile:

        nvcc cuda-edge-detect.cu -o cuda-edge-detect

To execute:

        ./cuda-edge-detect [threshold] < input > output

Example:

        ./cuda-edge-detect < BWstop-sign.pgm > BWstop-sign-edges.pgm

## Files

- [cuda-edge-detect.cu](cuda-edge-detect.cu) [hpc.h](hpc.h)
- [BWstop-sign.pgm](BWstop-sign.pgm)

***/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "hpc.h"
#include <string.h>

typedef struct {
    int width;   /* Width of the image (in pixels) */
    int height;  /* Height of the image (in pixels) */
    int maxgrey; /* Don't care (used only by the PGM read/write routines) */
    unsigned char *bmap; /* buffer of width*height bytes; each element represents the gray level of a pixel (0-255) */
} PGM_image;

const unsigned char WHITE = 255;
const unsigned char BLACK = 0;

/**
 * Initialize a PGM_image object: allocate space for a bitmap of size
 * `width` x `height`, and set all pixels to color `col`
 */
void init_pgm( PGM_image *img, int width, int height, unsigned char col )
{
    int i, j;

    assert(img != NULL);

    img->width = width;
    img->height = height;
    img->maxgrey = 255;
    img->bmap = (unsigned char*)malloc(width*height);
    assert(img->bmap != NULL);
    for (i=0; i<height; i++) {
        for (j=0; j<width; j++) {
            img->bmap[i*width + j] = col;
        }
    }
}

/**
 * Read a PGM file from file `f`. Warning: this function is not
 * robust: it may fail on legal PGM images, and may crash on invalid
 * files since no proper error checking is done.
 */
void read_pgm( FILE *f, PGM_image* img )
{
    char buf[1024];
    const size_t BUFSIZE = sizeof(buf);
    char *s;
    int nread;

    assert(f != NULL);
    assert(img != NULL);

    /* Get the file type (must be "P5") */
    s = fgets(buf, BUFSIZE, f);
    if (0 != strcmp(s, "P5\n")) {
        fprintf(stderr, "Wrong file type %s\n", buf);
        exit(EXIT_FAILURE);
    }
    /* Get any comment and ignore it; does not work if there are
       leading spaces in the comment line */
    do {
        s = fgets(buf, BUFSIZE, f);
    } while (s[0] == '#');
    /* Get width, height */
    sscanf(s, "%d %d", &(img->width), &(img->height));
    /* get maxgrey; must be less than or equal to 255 */
    s = fgets(buf, BUFSIZE, f);
    sscanf(s, "%d", &(img->maxgrey));
    if ( img->maxgrey > 255 ) {
        fprintf(stderr, "FATAL: maxgray=%d > 255\n", img->maxgrey);
        exit(EXIT_FAILURE);
    }
#if _XOPEN_SOURCE < 600
    img->bmap = (unsigned char*)malloc((img->width)*(img->height)*sizeof(unsigned char));
#else
    /* The pointer img->bmap must be properly aligned to allow aligned
       SIMD load/stores to work. */
    int ret = posix_memalign((void**)&(img->bmap), __BIGGEST_ALIGNMENT__, (img->width)*(img->height));
    assert( 0 == ret );
#endif
    assert(img->bmap != NULL);
    /* Get the binary data from the file */
    nread = fread(img->bmap, 1, (img->width)*(img->height), f);
    if ( (img->width)*(img->height) != nread ) {
        fprintf(stderr, "FATAL: error reading input: expecting %d bytes, got %d\n", (img->width)*(img->height), nread);
        exit(EXIT_FAILURE);
    }
}

/**
 * Write the image `img` to file `f`; if not NULL, use the string
 * `comment` as metadata.
 */
void write_pgm( FILE *f, const PGM_image* img, const char *comment )
{
    assert(f != NULL);
    assert(img != NULL);

    fprintf(f, "P5\n");
    fprintf(f, "# %s\n", comment != NULL ? comment : "");
    fprintf(f, "%d %d\n", img->width, img->height);
    fprintf(f, "%d\n", img->maxgrey);
    fwrite(img->bmap, 1, (img->width)*(img->height), f);
}

/**
 * Free the bitmap associated with image `img`; note that the
 * structure pointed to by `img` is NOT deallocated; only `img->bmap`
 * is.
 */
void free_pgm( PGM_image *img )
{
    assert(img != NULL);
    free(img->bmap);
    img->bmap = NULL; /* not necessary */
    img->width = img->height = img->maxgrey = -1;
}


#define BLOCK_SIZE 16
#define SHARED_SIZE (BLOCK_SIZE + 2)

__global__ void edge_detect_kernel(const unsigned char* in, unsigned char* edges, int width, int height, int threshold) {
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

    // Load corners 
    if (tx == 0 && ty == 0) smem[0][0] = (x > 0 && y > 0) ? in[(y - 1) * width + (x - 1)] : 0;
    if (tx == BLOCK_SIZE - 1 && ty == 0) smem[0][SHARED_SIZE - 1] = (x < width - 1 && y > 0) ? in[(y - 1) * width + (x + 1)] : 0;
    if (tx == 0 && ty == BLOCK_SIZE - 1) smem[SHARED_SIZE - 1][0] = (x > 0 && y < height - 1) ? in[(y + 1) * width + (x - 1)] : 0;
    if (tx == BLOCK_SIZE - 1 && ty == BLOCK_SIZE - 1) smem[SHARED_SIZE - 1][SHARED_SIZE - 1] = (x < width - 1 && y < height - 1) ? in[(y + 1) * width + (x + 1)] : 0;

    __syncthreads();

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int Gx = smem[ty][tx] - smem[ty][tx+2] 
               + 2 * smem[ty+1][tx] - 2 * smem[ty+1][tx+2] 
               + smem[ty+2][tx] - smem[ty+2][tx+2];
               
        int Gy = smem[ty][tx] + 2 * smem[ty][tx+1] + smem[ty][tx+2] 
               - smem[ty+2][tx] - 2 * smem[ty+2][tx+1] - smem[ty+2][tx+2];
               
        int magnitude = Gx * Gx + Gy * Gy;
        
        if (magnitude > threshold * threshold) {
            edges[y * width + x] = 255;
        } else {
            edges[y * width + x] = 0;
        }
    } else if (x < width && y < height) {
        edges[y * width + x] = 0; // or original pixel? the CPU code doesn't set edges on the boundary, leaving them untouched (or since initialized to WHITE, WHITE)
    }
}

void edge_detect_gpu(const PGM_image* in, PGM_image* edges, int threshold) {
    int img_size = in->width * in->height * sizeof(unsigned char);
    unsigned char *d_in, *d_edges;

    cudaSafeCall(cudaMalloc((void**)&d_in, img_size));
    cudaSafeCall(cudaMalloc((void**)&d_edges, img_size));

    cudaSafeCall(cudaMemcpy(d_in, in->bmap, img_size, cudaMemcpyHostToDevice));
    
    // Also copy the initialized WHITE bitmap to device so border pixels are correct if we don't modify them
    cudaSafeCall(cudaMemcpy(d_edges, edges->bmap, img_size, cudaMemcpyHostToDevice));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((in->width + BLOCK_SIZE - 1) / BLOCK_SIZE, (in->height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    edge_detect_kernel<<<grid, threads>>>(d_in, d_edges, in->width, in->height, threshold);
    cudaCheckError();

    cudaSafeCall(cudaMemcpy(edges->bmap, d_edges, img_size, cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_edges);
}

int main( int argc, char* argv[] )
{
    PGM_image bmap, out;
    int threshold = 70;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [threshold] < in.pgm > out.pgm\n", argv[0]);
        return EXIT_FAILURE;
    }
    if ( argc > 1 ) {
        threshold = atoi(argv[1]);
    }
    read_pgm(stdin, &bmap);
    init_pgm(&out, bmap.width, bmap.height, 255); // WHITE

    const double tstart = hpc_gettime();
    
    edge_detect_gpu(&bmap, &out, threshold);
    cudaDeviceSynchronize();
    
    const double elapsed = hpc_gettime() - tstart;
    fprintf(stderr, "CPU + GPU Execution time %.3f\n", elapsed);
    write_pgm(stdout, &out, "produced by cuda-edge-detect-gpu.cu");
    
    free_pgm(&bmap);
    free_pgm(&out);
    return EXIT_SUCCESS;
}
