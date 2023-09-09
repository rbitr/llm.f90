F16DIR = /home/andrew/code/scratch/fortran/scratch/sixteenbit/FP16/include/

convert.o: convert.c
	gcc -c -O3 -I$(F16DIR) convert.c -lm

llama2.o: llama2.f90
	gfortran -c -O3 -march=native -ffast-math -funroll-loops llama2.f90 -fopenmp

llm: convert.o llama2.o
	gfortran convert.o llama2.o -fopenmp -o llm

clean:
	rm *.o
