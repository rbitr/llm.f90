FORTRAN = gfortran-10
GCC = gcc-10

.DEFAULT_GOAL := all

weight_module.o: weight_module.f90 
	$(FORTRAN) -c -O3 -march=native -mtune=native -ffast-math -funroll-loops -flto -fPIC weight_module.f90

llama2.o: llama2.f90 
	$(FORTRAN) -c -O3 -march=native -mtune=native -ffast-math -funroll-loops -flto -fPIC llama2.f90  
read_ggml.o: read_ggml.f90
	$(FORTRAN) -c -O3 -march=native -mtune=native -ffast-math -funroll-loops -flto -fPIC read_ggml.f90

llm: weight_module.o read_ggml.o llama2.o 
	$(FORTRAN) -O3 -march=native -mtune=native -ffast-math -funroll-loops -flto -fPIC weight_module.o read_ggml.o llama2.o -o llm 

load: load.f90
	$(FORTRAN) -O3 -march=native -mtune=native -ffast-math -funroll-loops -flto -fPIC load.f90 -o load 
	

all: llm

clean:
	rm *.o
