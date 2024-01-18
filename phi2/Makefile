FORTRAN = gfortran-13
GCC = gcc-13

.DEFAULT_GOAL := all

FLAGS = -O3 -fprefetch-loop-arrays -march=native -mtune=native -ffast-math -funroll-loops -fno-strict-aliasing -flto -fwhole-program -fPIC 
 
#FLAGS = -O3 -fprefetch-loop-arrays -march=native -flto 

convert.o: convert.c
	$(GCC) -c $(FLAGS) -fno-strict-aliasing convert.c -lm 

weight_module.o: weight_module.f90 
	$(FORTRAN) -c $(FLAGS) weight_module.f90

phi2.o: phi2.f90 
	$(FORTRAN) -c $(FLAGS) phi2.f90  

read_ggml.o: read_ggml.f90 weight_module.o
	$(FORTRAN) -c $(FLAGS) read_ggml.f90

llm: weight_module.o read_ggml.o phi2.o convert.o
	$(FORTRAN) $(FLAGS) weight_module.o read_ggml.o phi2.o convert.o -o llm 


all: llm

clean:
	rm *.o
