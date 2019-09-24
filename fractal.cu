/*
Fractal code for CS 4380 / CS 5351

Copyright (c) 2019 Texas State University. All rights reserved.

Redistribution in source or binary form, with or without modification,
is *not* permitted. Use in source and binary forms, with or without
modification, is only permitted for academic use in CS 4380 or CS 5351
at Texas State University.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <sys/time.h>
#include "cs43805351.h"
#include <cuda.h>

static const int ThreadsPerBlock = 512;

static __global__ void fractalKernel(const int width, const int frames, unsigned char* const pic)
{
  const double Delta = 0.006;
  const double xMid = 0.232997;
  const double yMid = 0.550325;
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = idx % width;
  const int row = (idx / width) % width;
  const int frame = idx / (width * width);
  // compute frames
  if (frame < frames){// && col < width && row < width) {//check for in range
    const double delta = Delta * pow(0.985, frame);
    const double xMin = xMid - delta;
    const double yMin = yMid - delta;
    const double dw = 2.0 * delta / width;

    const double cy = yMin + row * dw;

        const double cx = xMin + col * dw;
        double x = cx;
        double y = cy;
        int depth = 256;
        double x2, y2;
        do {
          x2 = x * x;
          y2 = y * y;
          y = 2 * x * y + cy;
          x = x2 - y2 + cx;
          depth--;
        } while ((depth > 0) && ((x2 + y2) < 5.0));
        pic[frame * width * width + row * width + col] = (unsigned char)depth;
  }
}

static void CheckCuda(){
  cudaError_t e;
  cudaDeviceSynchronize();
  if(cudaSuccess != (e = cudaGetLastError())){
    fprintf(stderr, "CUDA error %d: %s\n", e, cudaGetErrorString(e));
    exit(-1);
  }
}

int main(int argc, char *argv[])
{
  printf("Fractal v1.8\n");

  // check command line
  if (argc != 3) {fprintf(stderr, "USAGE: %s frame_width num_frames\n", argv[0]); exit(-1);}
  const int width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "ERROR: frame_width must be at least 10\n"); exit(-1);}
  const int frames = atoi(argv[2]);
  if (frames < 1) {fprintf(stderr, "ERROR: num_frames must be at least 1\n"); exit(-1);}
  printf("frames: %d\n", frames);
  printf("width: %d\n", width);

  // allocate picture array
  unsigned char* pic = new unsigned char [frames * width * width];
  unsigned char* d_pic;//pic for device
  const int size = frames * width * width * sizeof(char);
  cudaMalloc((void **)&d_pic, size);//give size to device variable

  //copy data
  if(cudaSuccess != cudaMemcpy(d_pic, pic, size, cudaMemcpyHostToDevice)){fprintf(stderr, "copy to device failed"); exit(-1);}
  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // call timed function
  fractalKernel<<<((frames * width * width) + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(width, frames, d_pic);
  cudaDeviceSynchronize();

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.4f s\n", runtime);
  //copy data
  if(cudaSuccess != cudaMemcpy(pic, d_pic, size, cudaMemcpyDeviceToHost)){fprintf(stderr, "copy to host failed"); exit(-1);}
  CheckCuda();
  // write result to BMP files
  if ((width <= 256) && (frames <= 100)) {
    for (int frame = 0; frame < frames; frame++) {
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 1000);
      writeBMP(width, width, &pic[frame * width * width], name);
    }
  }
  cudaFree(d_pic);//free mem
  delete [] pic;
  return 0;
}
