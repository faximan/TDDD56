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

void array_swap(struct array* array, int ai, int bi) {
  int c = array_get(array, ai);
  array_insert(array, array_get(array, bi), ai);
  array_insert(array, c, bi);
}

int find_pivot_index(struct array* array, int from, int to) {
  return (from + to) / 2;
}

typedef struct {
  struct array* array;
  struct array* tmp;
  int pivot;
  int from;
  int to;
  int* left;
  int* right;

} par_partition_args;

void*
par_partition(void* vargs) {
  par_partition_args* args = (par_partition_args*)vargs;
  printf("par_partition pivot[%d] from[%d] to[%d] left[%d] right[%d]\n", args->pivot, args->from, args->to, *(args->left), *(args->right));
  int i;
  int cur;

  for (i = args->from; i < args->to; ++i) {
    cur = array_get(args->array, i);
    if (cur < args->pivot) {
      array_insert(args->tmp, cur, __sync_fetch_and_add(args->left, 1));
    } else {
      array_insert(args->tmp, cur, __sync_fetch_and_add(args->right, -1));
    }
  }

  pthread_exit(NULL);
}

// in place partitions array and returns the position of the pivot
int
seq_partition(struct array* array, int from, int to) {
  printf("seq_partition from[%d] to[%d]\n", from, to);
  int last = to - 1; // index of last element
  int pivot_index = find_pivot_index(array, from, to);
  int pivot = array_get(array, pivot_index);
  array_swap(array, pivot_index, last);

  int left = from;
  int i;
  for (i = from; i < last; ++i) {
    if (array_get(array, i) < pivot) {
      array_swap(array, i, left++);
    }
  }
  array_swap(array, last, left);

  return left;
}

int
start_par_partition(struct array* array, int from, int to, int num_threads) {
  return seq_partition(array, from, to);
  par_partition_args* args = malloc(num_threads * sizeof(par_partition_args));
  printf("start_par_partition from[%d] to[%d] num_threads[%d]\n", from, to, num_threads);

  pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
  pthread_attr_t attr;
  pthread_attr_init(&attr);

  int last = to - 1;

  struct array* tmp = array_alloc(array->length);
  tmp->length = to - from;
  int left = from;
  int right = last;
  int size = ((to - from) / num_threads);
  int pivot_index = find_pivot_index(array, from, to);
  int pivot = array_get(array, pivot_index);
  array_swap(array, pivot_index, last);

  int i;
  for (i = 0; i < num_threads; ++i) {
    args[i].array = array;
    args[i].tmp = tmp;
    args[i].pivot = pivot;
    args[i].from = (from + (i * size));
    if (i == (num_threads - 1)) {
      args[i].to = last;
    } else {
      args[i].to = (from + ((i + 1) * size));
    }
    args[i].left = &left;
    args[i].right = &right;

    pthread_create(&threads[i], &attr, &par_partition, (void*)&args[i]);
  }

  for (i = 0; i < num_threads; ++i) {
    pthread_join(threads[i], NULL);
  }
  array_insert(tmp, pivot, left);
  memcpy(array->data + from, tmp->data + from, sizeof(value) * (to - from + 1));

  return left;
}
//674 811 437 499 422 282 180 390 310 488

typedef struct {
  struct array* array;
  int from;
  int to;
  int threads;
} quicksort_args;

int
seq_quicksort(struct array* array, int from, int to) {
  if (from >= to) {
    return -1;
  }
  printf("seq_quicksort from[%d] to[%d]\n", from, to);
  /*
  printf("seq_partition from[%d] to[%d]\n", from, to);
  array_printf(array);
  printf("===============\n");
  */
  int mid = seq_partition(array, from, to);
  seq_quicksort(array, from, mid);
  seq_quicksort(array, mid + 1, to);

  return 0;
}

typedef struct {
  struct array* array;
  int from;
  int to;
  int level;
} par_quicksort_args;

void* quicksort(void* vargs);

int
par_quicksort(struct array* array, int from, int to, int level) {
  printf("par_quicksort from[%d] to[%d] level[%d]\n", from, to, level);

  int mid = start_par_partition(array, from, to, level);

  pthread_attr_t attr;
  pthread_attr_init(&attr);
  quicksort_args args[2];
  pthread_t threads[2];

  args[0].array = array;
  args[0].from = from;
  args[0].to = mid;
  args[0].threads = level - 2;
  args[1].array = array;
  args[1].from = mid + 1;
  args[1].to = to;
  args[1].threads = level - 2;

  int i;
  for (i = 0; i < 2; ++i) {
    pthread_create(&threads[i], &attr, quicksort, (void*)&args[i]);
  }
  for (i = 0; i < 2; ++i) {
    pthread_join(threads[i], NULL);
  }

  return 0;
}

void*
quicksort(void* vargs) {
  quicksort_args* args = (quicksort_args*)vargs;

  printf("quicksort from[%d] to[%d] threads[%d]\n", args->from, args->to, args->threads);

  if (args->from >= args->to) {
    return NULL;
  }

  if (args->threads <= 1) {
    seq_quicksort(args->array, args->from, args->to);
  } else {
    par_quicksort(args->array, args->from, args->to, args->threads);
  }

  pthread_exit(NULL);
}

int
quicksort_start(struct array* array) {
  int threads = NB_THREADS;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_t root_thread;

  par_quicksort_args args;
  args.array = array;
  args.from = 0;
  args.to = array->length;
  args.level = threads;

  pthread_create(&root_thread, &attr, quicksort, (void*)&args);
  pthread_join(root_thread, NULL);

  return 0;
}

int
sort(struct array * array)
{
  //seq_quicksort(array, 0, array->length);
  quicksort_start(array);
  //b = malloc(sizeof(value) * array->length);
  /*b = array_alloc(array->length);

  pthread_attr_init(&attr);

  par_quicksort(array, 0, array->length);
  */
  //printf("wut..\n");

  //simple_quicksort_ascending(array);

	return 0;
}

