// Matrix addition, CPU version
// gcc matrix_cpu.c -o matrix_cpu -std=c99

#include <stdio.h>

#define GRID_SIZE 64
#define BLOCK_SIZE 16
#define N 1024

__global__
void add_matrix(float *a, float *b, float *c)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int gsize = gridDim.x;
  int bsize = blockDim.x;
  int rowsize = gsize * bsize;
  int idx = (rowsize * (by * bsize + ty)) + (bx * bsize + tx);
  //int idx = (rowsize * (bx * bsize + tx)) + (by * bsize + ty);

  c[idx] = a[idx] + b[idx];
}

int main()
{
  float *a = new float[N*N];
  float *b = new float[N*N];
  float *c = new float[N*N];
  float* ad;
  float* bd;
  float* cd;

  cudaEvent_t beforeEvent;
  cudaEvent_t afterEvent;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++)	{
      a[i+j*N] = 10 + i;
      b[i+j*N] = (float)j / N;
    }
  }
  
  int size = N * N * sizeof(float);
  
  cudaMalloc((void**)&ad, size);
  cudaMalloc((void**)&bd, size);
  cudaMalloc((void**)&cd, size);

  dim3 dimBlock( BLOCK_SIZE, BLOCK_SIZE );
  dim3 dimGrid( GRID_SIZE, GRID_SIZE );

  cudaEventCreate(&beforeEvent);
  cudaEventCreate(&afterEvent);

  cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice); 
  cudaMemcpy(bd, b, size, cudaMemcpyHostToDevice);
  cudaEventRecord(beforeEvent, 0);
  add_matrix<<<dimGrid, dimBlock>>>(ad, bd, cd);
  cudaThreadSynchronize();

  cudaEventRecord(afterEvent, 0);
  cudaEventSynchronize(afterEvent);
  cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost ); 
  float theTime;
  cudaEventElapsedTime(&theTime, beforeEvent, afterEvent); 
  cudaFree( ad );
  cudaFree( bd );
  cudaFree( cd );

  printf("Total time in ms: %f\n", theTime);

 for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < N; j++)
  	{
  	  printf("%0.2f ", c[i+j*N]);
  	}
      printf("\n");
    }
  
  return EXIT_SUCCESS;
}
