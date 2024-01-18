## Fortran implementation of phi-2 llm

Should now work. Still needs some cleanup. You can get the .gguf model from TheBloke I believe or convert it yourself from the MS version released on HF.
Usage:

```
$ make
$ ./llm -m /mnt/ssd/llm/phi-2/ggml-model-f16.gguf -n 96 -t 0.9 -p "You can construct a Fibonacci sequence as follows:" -v
...

You can construct a Fibonacci sequence as follows:ĊĊ1. Give the first two numbers in the sequence as 0 and 1.Ċ2. For each consecutive number, calculate it by adding the two previous numbers together.Ċ3. The sequence will continue indefinitely, and the next number will be the sum of the last two numbers.ĊContinue the given sequence to generate the next four numbers: 0, 1, 1, 2, 3, 5, 8, 13 
 Inference time:    34.5540009      seconds
   2.74931979     tokens/second
 Timings
           1   84.4479141    
           2   1.04166670E-02
           3   3.88541675    
           4   258.218750    
           5   17.0208340  
# note that Ċ is a newline in the model vocabulary.

``` 
