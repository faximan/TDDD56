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
//#include "disable.h"

#ifndef DEBUG
#define NDEBUG
#endif

#include "pthread.h"
#include <string.h>
#include <stdlib.h>

#include "array.h"
#include "sort.h"
#include "simple_quicksort.h"

struct array* b;
pthread_attr_t attr;
pthread_t threads[NB_THREADS];

typedef struct {
  struct array* array;
  int pivot;
  int from;
  int to;
  int* left;
  int* right;

} partition_args;

void*
partition(void* args) {
  int i;
  int cur;
  partition_args* in = (partition_args*)args;

  //printf("from[%d] to[%d] pivot[%d]\n", in->from, in->to, in->pivot);
  //printf("partition [%d] [%d]\n", in->from, in->to);

  //printf("a kommer vara: %d ", in->array->length);
  //array_printf(in->array);
  //printf("========\n");

  for (i = in->from; i < in->to; ++i) {
    cur = array_get(in->array, i);
    //printf("partionerar in %d\n", cur);
    if (cur <= in->pivot) {
      //printf("mindre\n");
      array_insert(b, cur, __sync_fetch_and_add(in->left, 1));
    } else {
      //printf("större\n");
      array_insert(b, cur, __sync_fetch_and_add(in->right, -1));
    }
  }

  return NULL;
}

static int
ascending(const void* a, const void* b)
{
  int aa = *(value*) a;
  int bb = *(value*) b;

  return aa > bb;
}

int
par_quicksort(struct array* array, int first, int last) {

  if (last - first < 1000) {
    qsort(array->data, array->length, sizeof(value), ascending);
    return 0;
  }

  if (last - first < 2) {
    return 0;
  }
  //array_printf(array);

  //printf("par quick [%d] [%d]\n", first, last);

  int left = first;
  int right = last-1;

  int size = ((last - first) / NB_THREADS);

  int pivotIndex = (first + last) / 2;
  int pivot = array_get(array, pivotIndex);

  //int pivot = array_get(array, first);
  array_insert(array, array_get(array, last - 1), pivotIndex);
  array_insert(array, pivot, last - 1);

  //printf("pivot [%d]\n", pivot);

  //printf("first[%d] last[%d] pivot[%d] size[%d]\n", first, last, pivot, size);

  partition_args args[NB_THREADS];
  int i;
  for (i = 0; i < NB_THREADS; ++i) {
    args[i].array = array;
    args[i].pivot = pivot;
    args[i].from = (first + (i * size));
    if (i == NB_THREADS - 1) {
      args[i].to = last - 1;
    } else {
      args[i].to = (first + ((i + 1) * size));
    }
    args[i].left = &left;
    args[i].right = &right;

    pthread_create(&threads[i], &attr, &partition, (void*) &args[i]);
  }

  for (i = 0; i < NB_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }
  array_insert(b, pivot, left);

  //printf("left[%d] right[%d]\n", left, right);

  //printf("nu kommer b: %d", b->data[0]);
  //array_printf(b);
  //printf("===========\n");
  memcpy(array->data + first, b->data + first, sizeof(value) * (last - first));

  par_quicksort(array, first, left);
  par_quicksort(array, left + 1, last);

  return 0;
}

int
sort(struct array * array)
{
  //b = malloc(sizeof(value) * array->length);
  b = array_alloc(array->length);

  pthread_attr_init(&attr);

  par_quicksort(array, 0, array->length);

  //printf("wut..\n");

  //simple_quicksort_ascending(array);

	return 0;
}

