/*
 * stack.h
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

#include <stdlib.h>
#include <pthread.h>

#ifndef STACK_H
#define STACK_H

typedef struct element {
  int value;
  struct element* next;
} element_t;

typedef struct
{
  element_t* head;
#if NON_BLOCKING == 0
  pthread_mutex_t lock;
#endif
} stack_t;

// Allocates a stack.
stack_t * stack_alloc();
// Inits a stack. Should be run before using a stack.
// Returns 0 on success.
int stack_init(stack_t *);
// Pushes an element in a thread-safe manner
int stack_push_safe(stack_t *, element_t *);
// Pops an element in a thread-safe manner
int stack_pop_safe(stack_t *, element_t **);

#endif /* STACK_H */
