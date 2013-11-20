/*
 * stack.c
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
 *     but WITHOUT ANY WARRANTY without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef DEBUG
#define NDEBUG
#endif

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "stack.h"
#include "non_blocking.h"

#if NON_BLOCKING == 0
#warning Stacks are synchronized through locks
#else
#if NON_BLOCKING == 1 
#warning Stacks are synchronized through lock-based CAS
#else // NON_BLOCKING == 2
#warning Stacks are synchronized through hardware CAS
#endif
#endif

stack_t *
stack_alloc()
{
  stack_t *res;

  res = malloc(sizeof(stack_t));
  assert(res != NULL);

  if (res == NULL) {
    return NULL;
  }

  return res;
}

int
stack_init(stack_t *stack)
{
  assert(stack != NULL);
  stack->head = NULL;

#if NON_BLOCKING == 0
  pthread_mutexattr_t mutex_attr;
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutex_init(&stack->lock, &mutex_attr);
#endif
  return 0;
}

int
stack_check(stack_t *stack)
{
  /*** Optional ***/
  // Use code and assertions to make sure your stack is
  // in a consistent state and is safe to use.
  //
  // For now, just makes just the pointer is not NULL
  //
  // Debugging use only

  assert(stack != NULL);

  return 0;
}

int
stack_push_safe(stack_t *stack, element_t* new_element)
{
  assert(stack != NULL);
  
#if NON_BLOCKING == 0
  // Critical section, change head of stack.
  pthread_mutex_lock(&stack->lock);
  new_element->next = stack->head;
  stack->head = new_element;
  pthread_mutex_unlock(&stack->lock);

#elif NON_BLOCKING == 1
  /*** Optional ***/
  // Implement a harware CAS-based stack

#else  // NON_BLOCKING == 2
  element_t* old_element = NULL;
  do {
    old_element = stack->head;
    new_element->next = old_element;
  } while (cas((size_t *)&stack->head,
	       (size_t)old_element,
	       (size_t)new_element) != (size_t)old_element);
#endif
  return 0;
}

int
stack_pop_safe(stack_t *stack, element_t **popped_element)
{
#if NON_BLOCKING == 0
  pthread_mutex_lock(&stack->lock);
  assert(stack != NULL);
  *popped_element = stack->head;
  assert(*popped_element != NULL);
  stack->head = (*popped_element)->next;
  pthread_mutex_unlock(&stack->lock);
#elif NON_BLOCKING == 1
  /*** Optional ***/
  // Implement a harware CAS-based stack

#else  // NON_BLOCKING == 2
  element_t* old_head = NULL;

  do {
    old_head = stack->head;
  } while (cas((size_t *)&stack->head,
	       (size_t)old_head,
	       (size_t)old_head->next) != (size_t)old_head);
  *popped_element = old_head;
#endif
    return 0;
}

