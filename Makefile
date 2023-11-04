F16DIR = /home/andrew/code/scratch/fortran/scratch/sixteenbit/FP16/include/
FORTRAN = gfortran-10
GCC = gcc-10

.DEFAULT_GOAL := all

convert.o: convert.c
	$(GCC) -c -O3 -march=native -ffast-math -funroll-loops -flto convert.c -lm 

llama2.o: llama2.f90
	$(FORTRAN) -c -O3 -march=native -ffast-math -funroll-loops -flto -fopenmp  llama2.f90 

llm: convert.o llama2.o
	$(FORTRAN) -O3 -march=native -ffast-math -funroll-loops -flto -fopenmp convert.o llama2.o -o llm 

load.o: load.f90
	$(FORTRAN) -c load.f90

load: convert.o load.o
	$(FORTRAN) convert.o load.o -o load

all: load llm

clean:
	rm *.o
