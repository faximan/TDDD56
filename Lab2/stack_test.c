/*
 * stack_test.c
 *
 *  Created on: 18 Oct 2011
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

/* #ifndef DEBUG */
/* #define NDEBUG */
/* #endif */

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stddef.h>

#include "stack.h"
#include "non_blocking.h"

#define test_run(test)\
  printf("[%s:%s:%i] Running test '%s'... ", __FILE__, __FUNCTION__, __LINE__, #test);\
  test_setup();\
  if(test())\
  {\
    printf("passed\n");\
  }\
  else\
  {\
    printf("failed\n");\
  }\
  test_teardown();

typedef int data_t;
#define DATA_SIZE sizeof(data_t)
#define DATA_VALUE 5

stack_t *stack = NULL;
data_t data;

int do_push(int value) {
  element_t* elem = (element_t*)malloc(sizeof(element_t));
  elem->value = value;
  return stack_push_safe(stack, elem);
}

int do_pop(int* value) {
  element_t* elem = NULL;
  int status = stack_pop_safe(stack, &elem);
  if (status != 0) {
    return status;
  }
  *value = elem->value;
  free(elem);
  return 0;
}

void
test_init()
{
}

void
test_setup()
{
  // Allocate and initialize your test stack before each test
  data = DATA_VALUE;
  assert(stack == NULL);
  stack = stack_alloc();
  assert(stack_init(stack) == 0);
}

void
test_teardown()
{
  // Empty stack.
  while (stack->head != NULL) {
    int temp;
    do_pop(&temp);
  }
  free(stack);
  stack = NULL;
}

void
test_finalize()
{
}

int
test_push_safe()
{
  assert(stack != NULL);
  assert(stack->head == NULL);

  assert(do_push(5) == 0);
  int pushed = 0;
  assert(do_pop(&pushed) == 0);
  assert(pushed == 5);

  return 1;
}

void test_many_push() {
  int i = 0;  
  for (; i < MAX_PUSH_POP / NB_THREADS; i++) {
    do_push(DATA_VALUE);
  }
}



void test_many_pop() {
  int element;
  int i = 0;
  for (; i < MAX_PUSH_POP / NB_THREADS; i++) {
    do_pop(&element);
  }
}

// 3 Threads should be enough to raise and detect the ABA problem
#define ABA_NB_THREADS 3

int
test_aba()
{
  int success, aba_detected = 0;
  // Write here a test for the ABA problem
  success = aba_detected;
  return success;
}

// We test here the CAS function
struct thread_test_cas_args
{
  int id;
  size_t* counter;
  pthread_mutex_t *lock;
};
typedef struct thread_test_cas_args thread_test_cas_args_t;

void*
thread_test_cas(void* arg)
{
  thread_test_cas_args_t *args = (thread_test_cas_args_t*) arg;
  int i;
  size_t old, local;

  for (i = 0; i < MAX_PUSH_POP; i++)
    {
      do {
        old = *args->counter;
        local = old + 1;
      } while (cas(args->counter, old, local) != old);
    }

  return NULL;
}

int
test_cas()
{
#if 1
  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  thread_test_cas_args_t args[NB_THREADS];
  pthread_mutexattr_t mutex_attr;
  pthread_mutex_t lock;

  size_t counter;

  int i, success;

  counter = 0;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutex_init(&lock, &mutex_attr);

  for (i = 0; i < NB_THREADS; i++)
    {
      args[i].id = i;
      args[i].counter = &counter;
      args[i].lock = &lock;
      pthread_create(&thread[i], &attr, &thread_test_cas, (void*) &args[i]);
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }

  success = counter == (size_t)(NB_THREADS * MAX_PUSH_POP);

  if (!success)
    {
      printf("Got %ti, expected %i\n", counter, NB_THREADS * MAX_PUSH_POP);
    }

  assert(success);

  return success;
#else
  int a, b, c, *a_p, res;
  a = 1;
  b = 2;
  c = 3;

  a_p = &a;

  printf("&a=%X, a=%d, &b=%X, b=%d, &c=%X, c=%d, a_p=%X, *a_p=%d; cas returned %d\n", (unsigned int)&a, a, (unsigned int)&b, b, (unsigned int)&c, c, (unsigned int)a_p, *a_p, (unsigned int) res);

  res = cas((void**)&a_p, (void*)&c, (void*)&b);

  printf("&a=%X, a=%d, &b=%X, b=%d, &c=%X, c=%d, a_p=%X, *a_p=%d; cas returned %X\n", (unsigned int)&a, a, (unsigned int)&b, b, (unsigned int)&c, c, (unsigned int)a_p, *a_p, (unsigned int)res);

  return 0;
#endif
}

// Stack performance test
#if MEASURE != 0
struct stack_measure_arg
{
  int id;
};
typedef struct stack_measure_arg stack_measure_arg_t;

struct timespec t_start[NB_THREADS], t_stop[NB_THREADS], start, stop;

void*
test_push_pop(void *args) {
  stack_measure_arg_t *arg = (stack_measure_arg_t*) args;

#if MEASURE == 1
  clock_gettime(CLOCK_MONOTONIC, &t_start[arg->id]);
#endif
  test_many_push();

#if MEASURE == 2
  clock_gettime(CLOCK_MONOTONIC, &t_start[arg->id]);
  test_many_pop();
#endif

  clock_gettime(CLOCK_MONOTONIC, &t_stop[arg->id]);
  return NULL;
}

#endif

int
main(int argc, char **argv)
{
setbuf(stdout, NULL);
// MEASURE == 0 -> run unit tests
#if MEASURE == 0
  test_init();

  test_run(test_cas);

  test_run(test_push_safe);
  test_run(test_aba);

  test_finalize();
#else
  // Run performance tests
  int i;
  stack_measure_arg_t arg[NB_THREADS];  
  pthread_attr_t attr;
  pthread_t threads[NB_THREADS];

  pthread_attr_init(&attr);

  test_setup();
  clock_gettime(CLOCK_MONOTONIC, &start);

  for (i = 0; i < NB_THREADS; i++)
    {
      arg[i].id = i;
      pthread_create(&threads[i], &attr, &test_push_pop, (void*) &arg[i]);
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(threads[i], NULL);
    }

  // Wait for all threads to finish
  clock_gettime(CLOCK_MONOTONIC, &stop);

  // Print out results
  for (i = 0; i < NB_THREADS; i++)
    {
      printf("%i %i %li %i %li %i %li %i %li\n", i, (int) start.tv_sec,
          start.tv_nsec, (int) stop.tv_sec, stop.tv_nsec,
          (int) t_start[i].tv_sec, t_start[i].tv_nsec, (int) t_stop[i].tv_sec,
          t_stop[i].tv_nsec);
    }
#endif

  return 0;
}
