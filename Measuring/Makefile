#     Copyright 2011 Nicolas Melot
# 
#    Nicolas Melot (nicolas.melot@liu.se)
# 
# 
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
# 
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

FILES=compile compile-filled Makefile merge octave/* program.c settings time_difference_global.m Tutorial/report.pdf COPYING.txt matlab/* merge.m plot_data.m run run-filled start time_difference_thread.m variables variables-filled program_seq.c program_par.c program_both.c
ARCHIVE=Measuring.zip

NB_THREADS=1
ENTROPY=0.01
SUFFIX=

CFLAGS=-g -O0 -Wall -lrt -pthread -DMEASURE -DNB_THREADS=$(NB_THREADS) -DENTROPY=$(ENTROPY)

TARGET=program$(SUFFIX)

all: $(TARGET)

clean:
	$(RM) program
	$(RM) program-*
	$(RM) *.o
	$(RM) *.zip
	
$(TARGET): program.c
	gcc $(CFLAGS) -o program$(SUFFIX) program.c

.PHONY: all clean dist

dist:
	zip $(ARCHIVE) $(FILES)
