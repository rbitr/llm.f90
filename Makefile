
convert.o: convert.c
	gcc -c -O3 -I/home/andrew/code/scratch/fortran/scratch/sixteenbit/FP16/include/ convert.c -lm

llama2.o: llama2.f90
	gfortran -c -O3 -march=native -ffast-math -funroll-loops llama2.f90

llm: convert.o llama2.o
	gfortran convert.o llama2.o -o llm

clean:
	rm *.o
