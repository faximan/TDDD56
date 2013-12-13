/*
 * Rank sorting in sorting OpenCL
 * This kernel has a bug. What?
 */

__kernel void sort(__constant unsigned int *in_data, __global unsigned int *out_data, const unsigned int length)
{ 
  unsigned int pos = 0;
  unsigned int i;
  unsigned int val = in_data[get_global_id(0)];

  //find out how many values are smaller
  for (i = 0; i < length; i++)
    if (val > in_data[i])
      pos++;

  out_data[pos]=val;
}
