F16DIR = /home/andrew/code/scratch/fortran/scratch/sixteenbit/FP16/include/
FORTRAN = gfortran-10
GCC = gcc-10

convert.o: convert.c
	$(GCC) -c -O3 -I$(F16DIR) convert.c -lm

llama2.o: llama2.f90
	$(FORTRAN) -c -O3 -march=native -ffast-math -funroll-loops llama2.f90 -fopenmp

llm: convert.o llama2.o
	$(FORTRAN) convert.o llama2.o -fopenmp -o llm

clean:
	rm *.o
