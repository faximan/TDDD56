/*    Copyright 2011 Nicolas Melot
 *
 *   Nicolas Melot (nicolas.melot@liu.se)
 *
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define ENTROPY	0.1


static int
do_some_work(unsigned long long int count)
{
  unsigned long long int i;

  // Bring it on, yeah!
  for (i = 0; i < count; i++)
    ;

  return 0;
}

static unsigned long long int
variate(unsigned long long int count, float entropy)
{
  unsigned long long int variation, res;
  variation = (unsigned long long int) ((((float) (random() % RAND_MAX)) / RAND_MAX) * count
      * entropy);

  res = (unsigned long long int) (((float) (count + variation)));

  return res;
}

static int
do_work(unsigned long long int count, float entropy)
{
  count = variate(count, entropy);
  do_some_work(count);

  return 0;
}

int
main(int argc, char ** argv)
{
  unsigned long long int count;

  srandom(time(NULL));
  count = atoi(argv[1]);
  do_work(count, ENTROPY);

  return 0;
}
