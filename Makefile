FORTRAN = gfortran-10
GCC = gcc-10

.DEFAULT_GOAL := all

FLAGS = -O3 -march=native -mtune=native -ffast-math -funroll-loops -fno-strict-aliasing -flto -fPIC 
 
aligned_alloc.o: aligned_alloc.c
	$(GCC) -c $(FLAGS) aligned_alloc.c -lm

alignment_mod.o: alignment_mod.f90
	$(FORTRAN) -c $(FLAGS) alignment_mod.f90

convert.o: convert.c
	$(GCC) -c $(FLAGS)  convert.c -lm 

weight_module.o: weight_module.f90 
	$(FORTRAN) -c $(FLAGS) weight_module.f90

llama2.o: llama2.f90 
	$(FORTRAN) -c $(FLAGS) llama2.f90  

read_ggml.o: read_ggml.f90 weight_module.o alignment_mod.o
	$(FORTRAN) -c $(FLAGS) read_ggml.f90

llm: weight_module.o read_ggml.o llama2.o convert.o aligned_alloc.o alignment_mod.o
	$(FORTRAN) $(FLAGS) weight_module.o read_ggml.o convert.o llama2.o alignment_mod.o aligned_alloc.o -o llm 

load: load.f90
	$(FORTRAN) -O3 -march=native -mtune=native -ffast-math -funroll-loops -flto -fPIC load.f90 -o load 
	

all: llm

clean:
	rm *.o
