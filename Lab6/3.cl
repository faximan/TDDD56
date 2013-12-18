/*
 * Placeholder for wavelet transform.
 * Currently just a simple invert.
 */

#define dataWidth 512
#define dataHeight 512

__kernel void kernelmain(__global unsigned char *image,
			 __global unsigned char *data,
			 const unsigned int length)
{
  const int thread_id = get_global_id(0);
  const int block_id = thread_id / 3;
  const int offset = thread_id % 3;

  const int r = 2 * (block_id / (dataWidth / 2));
  const int c = 2 * (block_id % (dataWidth / 2));

  unsigned char in1, in2, in3, in4;
  unsigned char out1, out2, out3, out4;

  in1 = image[3 * (r * dataWidth + c) + offset];
  in2 = image[3 * (r * dataWidth + (c+1)) + offset];
  in3 = image[3 * ((r+1) * dataWidth + c) + offset];
  in4 = image[3 * ((r+1) * dataWidth + (c+1)) + offset];

  out1 = (in1 + in2 + in3 + in4) / 4;
  out2 = (in1 + in2 - in3 - in4) / 4 + 128;
  out3 = (in1 - in2 + in3 - in4) / 4 + 128;
  out4 = (in1 - in2 - in3 + in4) / 4 + 128;

  const int idx1 = 3 * ((r / 2) * dataWidth + (c / 2));
  const int idx2 = 3 * ((r / 2) * dataWidth + (c / 2) + (dataWidth / 2));
  const int idx3 = 3 * (((r / 2) + (dataHeight / 2)) * dataWidth + (c / 2));
  const int idx4 = 3 * (((r / 2) + (dataHeight / 2)) * dataWidth + (c / 2) + (dataWidth / 2));

  data[idx1 + offset] = out1;
  data[idx2 + offset] = out2;
  data[idx3 + offset] = out3;
  data[idx4 + offset] = out4;

}
