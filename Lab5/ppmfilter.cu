
#include <stdio.h>
#include "readppm.c"
#ifdef __APPLE__
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#else
#include <GL/glut.h>
#endif

#define BLOCKSIZE 16
#define GRIDSIZE 32

__global__ void filter(unsigned char *image, unsigned char *out, int n, int m)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int sumx, sumy, sumz, k, l;

  int row_size = blockDim.x + 4;
  int block_idx = (threadIdx.y + 2) * row_size + threadIdx.x + 2;
  
  __shared__ unsigned char s_image[3 * (BLOCKSIZE + 4) * (BLOCKSIZE + 4)];

  s_image[block_idx*3+0] = image[(i*n+j)*3+0];
  s_image[block_idx*3+1] = image[(i*n+j)*3+1];
  s_image[block_idx*3+2] = image[(i*n+j)*3+2];

  // Extra, if current index is on block border, pull in extra pixels.
  // Left.
  if (threadIdx.x < 2 && blockIdx.x != 0)
    {
      int offset = 2;
      s_image[ (block_idx - offset) * 3 + 0] = image[(i*n+j - offset) * 3 + 0];
      s_image[ (block_idx - offset) * 3 + 1] = image[(i*n+j - offset) * 3 + 1];
      s_image[ (block_idx - offset) * 3 + 2] = image[(i*n+j - offset) * 3 + 2];
    }
  // Right. 
  if (threadIdx.x >= blockDim.x - 2 && blockIdx.x != GRIDSIZE-1)
    {
      int offset = 2;
      s_image[ (block_idx + offset) * 3 + 0] = image[(i*n+j + offset) * 3 + 0];
      s_image[ (block_idx + offset) * 3 + 1] = image[(i*n+j + offset) * 3 + 1];
      s_image[ (block_idx + offset) * 3 + 2] = image[(i*n+j + offset) * 3 + 2];
    }
  // Up.
  if (threadIdx.y < 2 && blockIdx.y != 0)
    {
      int offset = 2;
      s_image[ (block_idx - offset*row_size) * 3 + 0] = image[(i*n+j - offset*n) * 3 + 0];
      s_image[ (block_idx - offset*row_size) * 3 + 1] = image[(i*n+j - offset*n) * 3 + 1];
      s_image[ (block_idx - offset*row_size) * 3 + 2] = image[(i*n+j - offset*n) * 3 + 2];
    }
  // Down.
  if (threadIdx.y >= blockDim.y - 2 && blockIdx.y != GRIDSIZE-1)
    {
      int offset = 2;
      s_image[ (block_idx + offset*row_size) * 3 + 0] = image[(i*n+j + offset*n) * 3 + 0];
      s_image[ (block_idx + offset*row_size) * 3 + 1] = image[(i*n+j + offset*n) * 3 + 1];
      s_image[ (block_idx + offset*row_size) * 3 + 2] = image[(i*n+j + offset*n) * 3 + 2];
    }

  // Corners.
  if (threadIdx.x < 2 && threadIdx.y < 2 && blockIdx.x != 0 && blockIdx.y != 0) {
    int block_offset = -2 * row_size - 2;
    int image_offset = -2 * n - 2;
    s_image[ (block_idx + block_offset) * 3 + 0 ] = image[(i*n+j + image_offset) * 3 + 0];
    s_image[ (block_idx + block_offset) * 3 + 1 ] = image[(i*n+j + image_offset) * 3 + 1];
    s_image[ (block_idx + block_offset) * 3 + 2 ] = image[(i*n+j + image_offset) * 3 + 2];
  }

  if (threadIdx.x < 2 && threadIdx.y >= blockDim.y - 2 && blockIdx.x != 0 && blockIdx.y != GRIDSIZE-1) {
    int block_offset = 2 * row_size - 2;
    int image_offset = 2 * n - 2;
    s_image[ (block_idx + block_offset) * 3 + 0 ] = image[(i*n+j + image_offset) * 3 + 0];
    s_image[ (block_idx + block_offset) * 3 + 1 ] = image[(i*n+j + image_offset) * 3 + 1];
    s_image[ (block_idx + block_offset) * 3 + 2 ] = image[(i*n+j + image_offset) * 3 + 2];
  }

  if (threadIdx.x >= blockDim.x - 2 && threadIdx.y < 2 && blockIdx.x != GRIDSIZE-1 && blockIdx.y != 0) {
    int block_offset = -2 * row_size + 2;
    int image_offset = -2 * n + 2;
    s_image[ (block_idx + block_offset) * 3 + 0 ] = image[(i*n+j + image_offset) * 3 + 0];
    s_image[ (block_idx + block_offset) * 3 + 1 ] = image[(i*n+j + image_offset) * 3 + 1];
    s_image[ (block_idx + block_offset) * 3 + 2 ] = image[(i*n+j + image_offset) * 3 + 2];
  }

  if (threadIdx.x >= blockDim.x - 2 && threadIdx.y >= blockDim.y - 2 &&
      blockIdx.x != GRIDSIZE-1 && blockIdx.y != GRIDSIZE-1) {
    int block_offset = 2 * row_size + 2;
    int image_offset = 2 * n + 2;
    s_image[ (block_idx + block_offset) * 3 + 0 ] = image[(i*n+j + image_offset) * 3 + 0];
    s_image[ (block_idx + block_offset) * 3 + 1 ] = image[(i*n+j + image_offset) * 3 + 1];
    s_image[ (block_idx + block_offset) * 3 + 2 ] = image[(i*n+j + image_offset) * 3 + 2];
  }
  
  __syncthreads();
  if (j < n && i < m)
    {
      out[(i*n+j)*3+0] = s_image[block_idx*3+0];
      out[(i*n+j)*3+1] = s_image[block_idx*3+1];
      out[(i*n+j)*3+2] = s_image[block_idx*3+2];
    }
  if (i > 1 && i < m-2 && j > 1 && j < n-2)
    {
      // Filter kernel
      sumx=0;sumy=0;sumz=0;
      for(k=-2;k<3;k++)
	for(l=-2;l<3;l++)
	  {
	    int index = ((threadIdx.y + 2 + k) * row_size + (threadIdx.x + 2 + l))*3;
	    sumx += s_image[index+0];
	    sumy += s_image[index+1];
	    sumz += s_image[index+2];
	  }
      out[(i*n+j)*3+0] = sumx/25;
      out[(i*n+j)*3+1] = sumy/25;
      out[(i*n+j)*3+2] = sumz/25;
    }
}


// Compute CUDA kernel and display image
void Draw()
{
  unsigned char *image, *out;
  int n, m;
  unsigned char *dev_image, *dev_out;
  
  image = readppm("maskros512.ppm", &n, &m);
  out = (unsigned char*) malloc(n*m*3);
  
  cudaMalloc( (void**)&dev_image, n*m*3);
  cudaMalloc( (void**)&dev_out, n*m*3);
  cudaMemcpy( dev_image, image, n*m*3, cudaMemcpyHostToDevice);
	
  dim3 dimBlock( BLOCKSIZE, BLOCKSIZE );
  dim3 dimGrid( GRIDSIZE, GRIDSIZE );

  cudaEvent_t beforeEvent;
  cudaEvent_t afterEvent;
  float theTime;
  cudaEventCreate(&beforeEvent);
  cudaEventCreate(&afterEvent);
  cudaEventRecord(beforeEvent, 0);
  
  filter<<<dimGrid, dimBlock>>>(dev_image, dev_out, n, m);
  cudaThreadSynchronize();
  cudaEventRecord(afterEvent, 0);
  cudaEventSynchronize(afterEvent);
  cudaEventElapsedTime(&theTime, beforeEvent, afterEvent); 
  printf("Time to draw: %f\n", theTime);
  
  cudaMemcpy( out, dev_out, n*m*3, cudaMemcpyDeviceToHost );
  cudaFree(dev_image);
  cudaFree(dev_out);
	
  // Dump the whole picture onto the screen.	
  glClearColor( 0.0, 0.0, 0.0, 1.0 );
  glClear( GL_COLOR_BUFFER_BIT );
  glRasterPos2f(-1, -1);
  glDrawPixels( n, m, GL_RGB, GL_UNSIGNED_BYTE, image );
  glRasterPos2i(0, -1);
  glDrawPixels( n, m, GL_RGB, GL_UNSIGNED_BYTE, out );
  glFlush();
}

// Main program, inits
int main( int argc, char** argv) 
{
  glutInit(&argc, argv);
  glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );
  glutInitWindowSize( 1024, 512 );
  glutCreateWindow("CUDA on live GL");
  glutDisplayFunc(Draw);
	
  glutMainLoop();
}
