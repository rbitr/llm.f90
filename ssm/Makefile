FORTRAN = gfortran-10
GCC = gcc-10

.DEFAULT_GOAL := all


FLAGS = -O2


weight_module.o: weight_module.f90 
	$(FORTRAN) -c $(FLAGS) weight_module.f90

mamba.o: mamba.f90 
	$(FORTRAN) -c $(FLAGS) mamba.f90  

llm: weight_module.o mamba.o 
	$(FORTRAN) $(FLAGS) weight_module.o mamba.o -o llm 
	
all: llm

clean:
	rm *.o

