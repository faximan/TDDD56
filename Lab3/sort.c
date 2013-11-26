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

pthread_attr_t attr;

int* b;

void
array_swap(struct array* array, const int ai, const int bi) {
  const int c = array->data[ai];
  array->data[ai] = array->data[bi];
  array->data[bi] = c;
}

// Returns index of pivot (median of first, middle, last)
int
get_pivot_index(struct array* array, const int lower, const int upper) {
  const int middle = (lower + upper) / 2;
  //  return middle;
  const int A = array->data[lower];
  const int B = array->data[middle];
  const int C = array->data[upper];
  
  if (A <= B) {
    if (C < A) {
      return lower;
    } else if (C > B) {
      return middle;
    } else {
      return upper;
    }
  } else {
    if (C < B) {
      return middle;
    } else if (C > A) {
      return lower;
    } else {
      return upper;
    }
  }
}

void
seq_selection_sort(struct array* array, const int from, const int to) {
  int i, j;
  for (i = from; i < to; i++) {
    int min_idx = i;
    int min_nbr = array->data[i];
    for (j = i + 1; j < to; j++) {
      if (array->data[j] < min_nbr) {
	min_nbr = array->data[j];
	min_idx = j;
      }
    }
    array_swap(array, i, min_idx);
  }
}

// In place partitions array and returns the position of the pivot.
int
seq_partition(struct array* array, const int from, const int to) {
  const int last = to - 1; // Index of last element.
  const int pivot_index = get_pivot_index(array, from, last);
  const int pivot = array->data[pivot_index];
  array_swap(array, pivot_index, last);

  int left = from;
  int i;
  for (i = from; i < last; ++i) {
    if ( (i%2) ?
	 array->data[i] < pivot :
	 array->data[i] <= pivot) {
      array_swap(array, i, left++);
    }
  }
  array_swap(array, last, left);

  return left;
}

typedef struct {
  struct array* array;
  int from;
  int to;
  int threads;
} quicksort_args;

void* quicksort(void* vargs);

void
seq_quicksort(struct array* array, const int from, const int to) {
  const int mid = seq_partition(array, from, to);
  quicksort_args args[2];

  args[0].array = array;
  args[0].threads = 1;
  args[0].from = from;
  args[0].to = mid;
  quicksort((void*)&args[0]);

  args[1].array = array;
  args[1].threads = 1;
  args[1].from = mid + 1;
  args[1].to = to;
  quicksort((void*)&args[1]);
}

inline int
fetch_and_add(int* variable, int value) {
  asm volatile( 
	       "lock; xaddl %%eax, %2;"
	       :"=a" (value)                   //Output
	       : "a" (value), "m" (*variable)  //Input
	       :"memory" );
  return value;
}

typedef struct {
  struct array* array;
  int from;
  int to;
  int pivot;

  int *left, *right;
} par_partition_args;


void*
par_partition_work(void* vargs) {
  par_partition_args* args = (par_partition_args*)vargs;

  int i, cur;
  for (i = args->from; i < args->to; i++) {
    cur = args->array->data[i];
    if ( (i%2) ? cur <= args->pivot : cur < args->pivot) {
      b[fetch_and_add(args->left, 1)] = cur;    
    } else {
      b[fetch_and_add(args->right, -1)] = cur;
    }
  }
  return args->left;
}

int
par_partition(struct array* array, int from, int to, int num_threads) {
  const int last = to - 1;
  const int pivot_index = get_pivot_index(array, from, last);
  const int pivot = array->data[pivot_index];
  array_swap(array, pivot_index, last);

  pthread_t threads[num_threads];
  par_partition_args args[num_threads];
  const int size = (to - from) / num_threads;

  int left = from;
  int right = to - 1;
  int i;
  for (i = 0; i < num_threads; i++) {
    args[i].array = array;
    args[i].from = from + (i * size);
    args[i].to = (i == num_threads - 1) ? to - 1 : from + ((i + 1) * size);
    args[i].pivot = pivot;
    args[i].left = &left;
    args[i].right = &right;
    pthread_create(&threads[i], &attr, par_partition_work, (void*)&args[i]);
  }

  for (i = 0; i < num_threads; i++) {
    pthread_join(threads[i], NULL);
  }

  b[left] = pivot;

  for (i = from; i < to; i++) {
    array->data[i] = b[i];
  }
  return left;
}

void
par_quicksort(struct array* array, int from, int to, int num_threads) {
  //const int mid = par_partition(array, from, to, num_threads);
  const int mid = seq_partition(array, from, to);

  quicksort_args args[2];
  const int num_threads_next_level = num_threads / 2;
  args[0].array = array;
  args[0].from = from;
  args[0].to = mid;
  args[0].threads = num_threads_next_level;
  args[1].array = array;
  args[1].from = mid + 1;
  args[1].to = to;
  args[1].threads = num_threads - num_threads_next_level; // All that are left.

  pthread_t new_thread;
  pthread_create(&new_thread, &attr, quicksort, (void*)&args[0]);
  quicksort((void*)&args[1]);
  pthread_join(new_thread, NULL);
}

void*
quicksort(void* vargs) {
  const quicksort_args* args = (quicksort_args*)vargs;

  if (args->to - args->from <= 1) {
    return NULL;
  } else if (args->to - args->from <= 10) {
    seq_selection_sort(args->array, args->from, args->to);
  } else if (args->threads <= 1) {
    seq_quicksort(args->array, args->from, args->to);
  } else {
    par_quicksort(args->array, args->from, args->to, args->threads);
  }
  return NULL;
}

int
quicksort_start(struct array* array) {
  pthread_attr_init(&attr);

  quicksort_args args;
  args.array = array;
  args.from = 0;
  args.to = array->length;
  args.threads = NB_THREADS;

  b = malloc(array->length * sizeof(int));
  quicksort((void*)&args);
  free(b);

  return 0;
}

int
sort(struct array * array)
{
  quicksort_start(array);
  //simple_quicksort_ascending(array);
  return 0;
}

