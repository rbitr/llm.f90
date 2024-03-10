## Fortran implementation of phi-2 llm

Phi-2 is a model developed by Microsoft. See https://huggingface.co/microsoft/phi-2

Most notably, the weights are licensed as Apache 2.0, meaning this is a true open source model not encumbered by any proprietary restrictions like Meta's llama. As such, in my view it's the preferred model to work with right now, if it's suitable powerful. The model is "small" at 2.7 G parameters, but supposedly punches above its weight. 

You can see it is comparatively fast, on my (not so good) computer it runs at 2.75 tok/s on one CPU core (though I think it doesn't scale well to more cores due to memory bandwidth). On a modern computer it should easily be fast enough to use in real time. Part of the [llm.f90](https://github.com/rbitr/llm.f90) model family. __Even if you don't want to run Fortran code, this is a simple and dependency free minimal implementation that makes it comparatively easy to implement the model in your language of choice.__

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

Now includes proper(ish) unicode support. Tested with the [phi2-ko](https://huggingface.co/daekeun-ml/phi-2-ko-v0.1) Korean model:

```
$ wget https://huggingface.co/SDFASDGA/llm/resolve/main/phi-2-ko-v0.1/ggml-model-f16.gguf
...
$ ./llm -m /mnt/ssd/llm/phi-2-ko-v0.1/ggml-model-f16.gguf -n 96 -t 0.9 -p "시도해  볼만한 음식 목록은 다음과 같습니다." -v
found 66676 tokens
 loading merges
found 66529 merges
 maximum token length          256
 Loaded weights
 Pre-tokenized prompt: ìĭľëıĦíķ´Ġë³¼ë§ĮíķľĠìĿĮìĭĿĠëª©ë¡ĿìĿĢĠëĭ¤ìĿĮê³¼Ġê°ĻìĬµëĭĪëĭ¤.
       62205 ìĭľëıĦ                                                    
       50322 íķ´                                                          
       51308 Ġë³¼                                                        
       52797 ë§Įíķľ                                                    
       55763 ĠìĿĮìĭĿ                                                  
       54401 Ġëª©ë¡Ŀ                                                  
       50299 ìĿĢ                                                          
       53962 Ġëĭ¤ìĿĮê³¼                                            
       60150 Ġê°ĻìĬµëĭĪëĭ¤                                      
          14 .                                                               
시도해 볼만한 음식 목록은 다음과 같습니다.

* 랍큐'나쿠사: 닭고기의 닭다리살, 해삼, 카레소스를 넣어 끓인 햄.
* 라디에이터 스칸다르: 감자튀김, 토마토 소스, 양파, 손질한 감자를 넣어 튀김.
* 미카야 이투리: 돼지고기, 유부갈비, 야채, 허브 등을 조리가 
 Inference time:    34.4960022      seconds
   2.75394249     tokens/second
 Timings
           1   70.6666641    
           2   0.00000000    
           3   3.33333325    
           4   270.000000    
           5   18.6666660    

```

Implementation note: phi appears to differ from llama in that the rotary encoding is only applied to part of the q/k values (32/80 on each head) and that the real/imag parts are grouped and concatenated instead of alternating. Once you figure that out, it's easy to implement.
