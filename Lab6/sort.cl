
#define MIN(a,b) (((a)<(b))?(a):(b))

#define NUM_SHARED_ELEMENTS 512

__kernel void sort(__constant unsigned int *in_data, __global unsigned int *out_data, const unsigned int length)
{ 
  unsigned int pos = 0;
  unsigned int val = in_data[ get_global_id(0) ];

  __local unsigned int shared_data[NUM_SHARED_ELEMENTS];

  int thread_id = get_local_id(0) % NUM_SHARED_ELEMENTS;

  int num_blocks = (length / NUM_SHARED_ELEMENTS) + 1;

  for (int i = 0; i < num_blocks; ++i) {
    int offset = i * NUM_SHARED_ELEMENTS;

    if (thread_id + offset < length) {
      shared_data[thread_id] = in_data[thread_id + offset]; 
    }

    barrier(CLK_GLOBAL_MEM_FENCE);  // Global?

    // Find out how many values are smaller for this chunk of numbers.
    for (int k = 0; k < MIN(NUM_SHARED_ELEMENTS, length - offset); k++) {
      if (val > shared_data[k]) {
	pos++;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  out_data[pos] = val;
}


