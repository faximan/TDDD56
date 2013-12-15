/*
 * sort.c
 *
 *  Created on: 5 Sep 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 *
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 *
 */

// Do not touch or move these lines
#include <stdio.h>
#include "disable.h"

#ifndef DEBUG
#define NDEBUG
#endif

#include "pthread.h"
#include <string.h>
#include <stdlib.h>

#include "array.h"
#include "sort.h"
#include "simple_quicksort.h"

int threshold;
pthread_attr_t attr;

struct array* array;
int* temp_array[NB_THREADS];
int current_index[NB_THREADS];
int pivots[NB_THREADS - 1];

typedef struct {
  int from, to;
  int start;
  int* array;
} parallel_args;

inline void swap(int* array, int ai, int bi) {
  const int c = array[ai];
  array[ai] = array[bi];
  array[bi] = c;
}

void seq_selection_sort(int* array, const int from, const int to) {
  int i, j;
  for (i = from; i < to; i++) {
    int min_idx = i;
    int min_nbr = array[i];
    for (j = i + 1; j < to; j++) {
      if (array[j] < min_nbr) {
	min_nbr = array[j];
	min_idx = j;
      }
    }
    swap(array, i, min_idx);
  }
}

void seq_bubble_sort(int* array, const int from, const int to) {
  int i;
  int done;
  do {
    done = 1;
    for (i = from + 1; i < to; i++) {
      if (array[i] < array[i-1]) {
	swap(array, i, i-1);
	done = 0;
      }
    }
  } while (!done);
}

inline int seq_find_pivot(int* array, const int from, const int to) {
  const int mid = (from + to) / 2;

  const int A = array[mid-1];
  const int B = array[mid];
  const int C = array[mid+1];
  if (A > B) {
    if (B > C) return mid;
    if (A > C) return mid+1;
    return mid-1;
  }
  if (A > C) return mid-1;
  if (B > C) return mid+1;
  return mid;
}

// In place partitions array and returns the position of the pivot.
int seq_partition(int* array, const int from, const int to) {
  const int last = to - 1;  // Index of last element.
  const int pivot_index = seq_find_pivot(array, from, last);
  const int pivot = array[pivot_index];
  swap(array, pivot_index, last);

  int left = from;
  int i;
  for (i = from; i < last; ++i) {
    // To balance cases when elements are equal to pivot.
    if ( (i%2) ?
	 array[i] <= pivot :
	 array[i] < pivot) {
      swap(array, i, left++);
    }
  }
  swap(array, last, left);
  return left;
}

void seq_quicksort(int* array, const int from, const int to) {
  if (to - from < threshold) {
    seq_selection_sort(array, from, to);
    //seq_bubble_sort(array, from, to);
    return;
  }

  const int mid = seq_partition(array, from, to);
  seq_quicksort(array, from, mid);
  seq_quicksort(array, mid+1, to);
}

void* seq_quicksort_start(void* vargs) {
  parallel_args* args = (parallel_args*)vargs;
  seq_quicksort(args->array, args->from, args->to);

  int i;
  for (i = args->from; i < args->to; i++) {
    array->data[i + args->start] = args->array[i];
  }
  return NULL;
}

// Thread-safe fetch and increment operation for shared variables.
inline int fetch_and_add(int* variable, int value) {
  asm volatile(
	       "lock; xaddl %%eax, %2;"
	       :"=a" (value)
	       : "a" (value), "m" (*variable)
	       :"memory" );
  return value;
}

// Loops through the array to be sorted and puts every element in the right sub help array based on its relation to the pivots.
void* fast_sort_partition(void* vargs) {
  parallel_args* args = (parallel_args*)vargs;
  int i;
  for (i = args->from; i < args->to; i++) {
    int j = 0;
    const int cur = args->array[i];
    const int offset = i % NB_THREADS;
    while (cur > pivots[j] || (cur == pivots[j] && offset > j)) {
      ++j;
      if (j >= NB_THREADS - 1) {
	break;
      }
    }
    temp_array[j][fetch_and_add(&current_index[j], 1)] = cur;
  }
  return NULL;
}

// Selects NBTHREADS-1 pivots from the input array in a good way, sorts them and puts them in the pivot array.
void find_pivots() {
  const int num_pivots = NB_THREADS - 1;
  const int sample_size = num_pivots + (6 * (num_pivots + 1));
  
  // Start by sampling values.
  int candidates[sample_size];
  
  const int sample_interval = array->length / sample_size;
  const int mid = sample_interval / 2;
  int i;
  for (i = 0; i < sample_size; i++) {
    candidates[i] = array->data[i * sample_interval + mid];
  }
  // Sort them and pick pivots not close to each other.
  seq_quicksort(candidates, 0, sample_size);

  for (i = 0; i < num_pivots; i++) {
    pivots[i] = candidates[(i * 7) + 6];
  }
}

int sort(struct array *my_array)
{
  threshold = 12;
  pthread_attr_init(&attr);
  array = my_array;

  // Do not include the pivots in the array to be sorted.
  const int length = array->length;
  const int size = length / NB_THREADS;
  int i;
  for (i = 0; i < NB_THREADS; i++) {
    // Worst case, all elements will end up in the same
    // temporary array.
    temp_array[i] = malloc(array->length * sizeof(int));
    current_index[i] = 0;
  }

  find_pivots();

  // Split the partitioning equally on all threads.
  pthread_t threads[NB_THREADS];
  parallel_args args[NB_THREADS];
  for (i = 0; i < NB_THREADS; i++) {
    args[i].from = i * size;
    args[i].to = (i == NB_THREADS - 1) ? length : (i + 1) * size;
    args[i].array = array->data;
    pthread_create(&threads[i], &attr, fast_sort_partition, (void*)&args[i]);
  }
  for (i = 0; i < NB_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }

  // Copy elements back to original parallely.
  int start = 0;
  for (i = 0; i < NB_THREADS; i++) {
    args[i].from = 0;
    args[i].to = current_index[i];
    args[i].start = start;
    args[i].array = temp_array[i];
    pthread_create(&threads[i], &attr, seq_quicksort_start, (void*)&args[i]);
    start += current_index[i];
  }
  for (i = 0; i < NB_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }
  return 0;
}

