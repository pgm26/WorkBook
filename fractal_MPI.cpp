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
#include <mpi.h>

static void fractal(const int width, const int frames, unsigned char* const pic, const long start, const long end)
{
  const double Delta = 0.006;
  const double xMid = 0.232997;
  const double yMid = 0.550325;

  //mark loop with start and end areas
  for (int frame = start; frame < end; frame++) {  // frames
    const double delta = Delta * pow(0.985, frame);
    const double xMin = xMid - delta;
    const double yMin = yMid - delta;
    const double dw = 2.0 * delta / width;
    for (int row = 0; row < width; row++) {  // rows
      const double cy = yMin + row * dw;
      for (int col = 0; col < width; col++) {  // columns
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
  }
}

int main(int argc, char *argv[])
{
  //MPI start
  int comm_sz, my_rank;
  MPI_Init(NULL,NULL);//setup MPI
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  if(my_rank == 0){printf("Fractal v1.8\n");}//only process 0 prints
  // check command line
  if (argc != 3) {fprintf(stderr, "USAGE: %s frame_width num_frames\n", argv[0]); exit(-1);}
  const int width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "ERROR: frame_width must be at least 10\n"); exit(-1);}
  const int frames = atoi(argv[2]);
  if (frames < 1) {fprintf(stderr, "ERROR: num_frames must be at least 1\n"); exit(-1);}
  if (frames % comm_sz != 0){
      fprintf(stderr, "ERROR: number of frames isn't a multiple of the processes\n");
      exit(-1);
  }//error statement for number of frames being a part of the number of processes
  if(my_rank == 0){printf("frames: %d\n", frames);}
  if(my_rank == 0){printf("width: %d\n", width);}
  const int size = frames*width*width;//size of pic
  // allocate picture array
  unsigned char* pic = new unsigned char [size];
  unsigned char* finish = new unsigned char [size];

  //compute bounds for block partition
  const long my_start = my_rank * frames / comm_sz;
  const long my_end = (my_rank + 1) * frames / comm_sz;
  // start time
  timeval start, end;
  MPI_Barrier(MPI_COMM_WORLD);//everyone starts the timer at the same time
  gettimeofday(&start, NULL);

  // call timed function
  fractal(width, frames, pic, my_start, my_end);
  MPI_Gather(pic, (frames/comm_sz*width*width),
    MPI_UNSIGNED_CHAR, finish, (frames/comm_sz * width *width), MPI_CHAR, 0, MPI_COMM_WORLD);
  //gather data into one array
  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  if(my_rank == 0){printf("compute time: %.4f s\n", runtime);}

  // write result to BMP files only if my rank is 0
  if ((width <= 256) && (frames <= 100) && my_rank == 0) {
    for (int frame = 0; frame < frames; frame++) {
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 1000);
      writeBMP(width, width, &finish[frame * width * width], name);
    }
  }
//finalize MPI
  MPI_Finalize();
  delete [] finish;//clean up dynamic memory
  delete [] pic;
  return 0;
}