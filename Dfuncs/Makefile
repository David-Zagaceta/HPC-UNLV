SHELL='bash'
#
# Bridges - PSC
#
# Intel Compilers are loaded by default
# You will need to specifically switch to GNU Modules
# With with `modules.sh`
#

CC = g++
OPENMP = -fopenmp
CFLAGS = -O3 -std=c++11
LIBS =


TARGETS = Poly Recursive

all:	$(TARGETS)

Poly: Poly.o factorial.o
	$(CC) -o $@ $(LIBS) $(OPENMP)  Poly.o factorial.o
Recursive: Recursive.o factorial.o
	$(CC) -o $@ $(LIBS) Recursive.o factorial.o

Poly.o: Poly.cpp factorial.h
	$(CC) -c $(OPENMP) $(CFLAGS) Poly.cpp
Recursive.o: Recursive.cpp factorial.h
	$(CC) -c $(CFLAGS) Recursive.cpp
factorial.o: factorial.cpp factorial.h
	$(CC) -c $(CFLAGS) factorial.cpp

clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
