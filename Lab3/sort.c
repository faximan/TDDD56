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

#define THRESHOLD 10

pthread_attr_t attr;

struct array* array;
int* b[NB_THREADS];
int current_index[NB_THREADS];
int pivots[NB_THREADS-1];

typedef struct {
  int from, to;
  int start;
  int* array;
} parallel_args;

void swap(int* array, int ai, int bi) {
  const int c = array[ai];
  array[ai] = array[bi];
  array[bi] = c;
}

// Returns index of pivot (median of first, middle, last)
int
get_pivot_index(int* array, const int lower, const int upper) {
  int middle = (lower + upper) / 2;
  return middle;
}

void
seq_selection_sort(int* array, const int from, const int to) {
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

// In place partitions array and returns the position of the pivot.
int
seq_partition(int* array, const int from, const int to) {
  const int last = to - 1; // Index of last element.
  const int pivot_index = get_pivot_index(array, from, last);
  const int pivot = array[pivot_index];
  swap(array, pivot_index, last);

  int left = from;
  int i;
  for (i = from; i < last; ++i) {
    if ( (i%2) ?
	 array[i] <= pivot :
	 array[i] < pivot) {
      swap(array, i, left++);
    }
  }
  swap(array, last, left);
  return left;
}

void
seq_quicksort(int* array, const int from, const int to) {
  if (to - from < THRESHOLD) {
    seq_selection_sort(array, from, to);
    return;
  }

  const int mid = seq_partition(array, from, to);
  seq_quicksort(array, from, mid);
  seq_quicksort(array, mid+1, to);
}

void*
seq_quicksort_start(void* vargs) {
  parallel_args* args = (parallel_args*)vargs;
  seq_quicksort(args->array, args->from, args->to);

  int i;
  for (i = args->from; i < args->to; i++) {
    array->data[i + args->start] = args->array[i];
  }
  return NULL;
}

int
fetch_and_add(int* variable, int value) {
  asm volatile(
	       "lock; xaddl %%eax, %2;"
	       :"=a" (value)                   //Output
	       : "a" (value), "m" (*variable)  //Input
	       :"memory" );
  return value;
}

void*
fast_sort_partition(void* vargs) {
  parallel_args* args = (parallel_args*)vargs;
  int i;
  for (i = args->from; i < args->to; i++) {
    int j = 0;
    int cur = args->array[i];
    while (cur >= pivots[j]) {
      ++j;
      if (j >= NB_THREADS - 1) {
	break;
      }
    }
    b[j][fetch_and_add(&current_index[j], 1)] = cur;
  }

  return NULL;
}

int
fast_sort_start(struct array* my_array) {
  pthread_attr_init(&attr);
  array = my_array;

  const int length = array->length - (NB_THREADS - 1);
  const int size = length / NB_THREADS;
  int i;
  for (i = 0; i < NB_THREADS; i++) {
    b[i] = malloc(array->length * sizeof(int));
    current_index[i] = 0;
    if (i != NB_THREADS-1)
      swap(array->data, length+i, size * (i + 1));
  }

  for (i = 0; i < NB_THREADS-1; i++) {
    pivots[i] = array->data[array->length - 1 - i];
  }
  seq_selection_sort(pivots, 0, NB_THREADS-1);

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
  for (i = 0; i < NB_THREADS - 1; i++) {
    b[i][current_index[i]++] = pivots[i];
  }

  /* for (i = 0; i < NB_THREADS; i++) { */
  /*   printf("b[%d]: \n", i); */
  /*   int j; */
  /*   for (j = 0; j < current_index[i]; j++) { */
  /*     printf("%d ", b[i][j]); */
  /*   } */
  /*   printf("\n=================\n"); */
  /* } */


  int start = 0;
  for (i = 0; i < NB_THREADS; i++) {
    args[i].from = 0;
    args[i].to = current_index[i];
    args[i].start = start;
    args[i].array = b[i];
    pthread_create(&threads[i], &attr, seq_quicksort_start, (void*)&args[i]);
    start += current_index[i];
  }
  for (i = 0; i < NB_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }
  return 0;
}

int
sort(struct array * array)
{
#if NB_THREADS != 1
  fast_sort_start(array);
#else
  seq_quicksort(array->data, 0, array->length);
#endif
  //simple_quicksort_ascending(array);
  return 0;
}

