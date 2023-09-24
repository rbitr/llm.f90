F16DIR = /home/andrew/code/scratch/fortran/scratch/sixteenbit/FP16/include/
FORTRAN = gfortran-10
GCC = gcc-10
EXEFILE = llama_q4

convert.o: convert.c
	$(GCC) -c -O3 -I$(F16DIR) convert.c -lm

llama2.o: $(EXEFILE).f90
	$(FORTRAN) -c -O3 -march=native -ffast-math -funroll-loops -fopenmp $(EXEFILE).f90 

llm: convert.o llama2.o
	$(FORTRAN) convert.o $(EXEFILE).o -fopenmp -o llm

clean:
	rm *.o
