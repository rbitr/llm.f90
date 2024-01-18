## Fortran implementation of phi-2 llm

This is a "small" LLM that supposedly peforms well. You can see it is comparatively fast, on my (not so good) computer it runs at 2.75 tok/s on one CPU core (though I think it doesn't scale well to more cores due to memory bandwidth). Part of the [llm.f90](https://github.com/rbitr/llm.f90) model family. __Even if you don't want to run Fortran code, this is a simple and dependency free minimal implementation that makes it comparatively easy to implement the model in your language of choice.__

Still needs some cleanup. I'm hosting the model in gguf format on HF. You can also get the .gguf model from TheBloke I believe or convert it yourself from the MS version released on HF.

Current version uses AVX2 to convert from fp16 and for dot-products. For non-Intel it will require modification which should be straightforward, I can help you if there are any problems. 

Usage:

```
$ git clone https://github.com/rbitr/llm.f90
$ git switch dev/phi2 # the right branch
$ cd phi2
$ wget https://huggingface.co/SDFASDGA/llm/resolve/main/phi2-ggml-model-f16.gguf # model weights. 2.7B
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

Implementation note: phi appears to differ from llama in that the rotary encoding is only applied to part of the q/k values (32/80 on each head) and that the real/imag parts are grouped and concatenated instead of alternating. Once you figure that out, it's easy to implement.