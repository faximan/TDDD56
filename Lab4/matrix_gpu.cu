// Matrix addition, CPU version
// gcc matrix_cpu.c -o matrix_cpu -std=c99

#include <stdio.h>

#define GRID_SIZE 2
#define BLOCK_SIZE 4
#define N 16

__global__
void add_matrix(float *a, float *b, float *c)
{
  //int idx = blockIdx.x * blockDim.x + threadIdx.x * blockDim.x + threadIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int gsize = gridDim.x;
  int bsize = blockDim.x;
  int rowsize = gsize * bsize;
  

  int idx = (rowsize * (by * bsize + ty)) + (bx * bsize + tx);

  

  c[idx] = a[idx] + b[idx];
}

int main()
{
  float a[N*N];
  float b[N*N];
  float c[N*N];
  float* ad;
  float* bd;
  float* cd;

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
  cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice); 
  cudaMemcpy(bd, b, size, cudaMemcpyHostToDevice);
  add_matrix<<<dimGrid, dimBlock>>>(ad, bd, cd);
  cudaThreadSynchronize();
  cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost ); 
  cudaFree( ad );
  cudaFree( bd );
  cudaFree( cd );

  //  printf("%f\n", c[2]);

  
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
