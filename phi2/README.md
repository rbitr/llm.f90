## Fortran implementation of phi-2 llm

Still under development. Runs but does not generate correct output after the first few tokens.

Usage:

```
$ make
$ ./llm -m /mnt/ssd/llm/phi-2/ggml-model-f16.gguf -n 96 -t 0.9 -p "I stopped posting on knitting forums because" -v
``` 
