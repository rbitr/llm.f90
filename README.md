# llama2.f90
LLaMA2 model inference in Fortran

## About

Supports llama2 models in gguf format (f16 only - but see the four\_bit\_dev branch)

For Fortran implementations of inference for various GPT models see https://github.com/certik/fastgpt

## Use

### Dependencies
This unfortunately uses a separate c-library somewhat trivially to convert between f32 and f16. 
```bash
git clone https://github.com/Maratyszcza/FP16/
```

### Clone the repo and build
```bash
git clone https://github.com/rbitr/llama2.f90
cd llama2.f90
```
Edit the `Makefile` to add the path of FP16/include cloned above. Then make

```bash
make
```

### Get a model file, or convert a pytorch model with llama.cpp:

```bash
wget https://huggingface.co/SlyEcho/open_llama_3b_v2_gguf/resolve/main/open-llama-3b-v2-f16.gguf
```

### Convert to compatible format
```bash
./load -i <model file you downloaded> -o <output file> -t <output vocab file>
```
Notes: 

1. The programs could be trivially changed to read ggml directly, but it would look more complicated

2. If you have another vocab file it may work better. The ggml files use unicode as a placehodler for spaces and it's currently handled in a hacky way.

### Run the model

```bash
./llm -m ./models/llama3b.bin -s ./models/3btokens.bin -v -t 0.9 -p "I stopped posting in knitting forums becuase" -n 128
 Embedding dimension:         3200
 Hidden dimension:         8640
 Layers:           26
 Heads:           32
 kv Heads:           32
 Vocabulary Size:       -32000
 Sequence Length:         2048
 loaded embedding weights:   102400000
 loaded rms att weights:       83200
 loaded wq weights:   266240000
 loaded wk weights:   266240000
 loaded wv weights:   266240000
 loaded wo weights:   266240000
 loaded rms ffn  weights:       83200
 loaded w1 weights:   718848000
 loaded w2 weights:   718848000
 loaded w3 weights:   718848000
 loaded rms_final weights:        3200
 loaded freq cis real  weights:      102400
 loaded freq_cis_imag weights:      102400
 loaded wcls weights:   102400000
 Loaded weights
I stopped posting in knitting forums becuase I felt like everything I posted was either old or everyone knew about it. I didn't want to talk about the same things over and over and over again. So now I post on my blog and then feel like I can talk about whatever I want here and no one's gonna know about it. It's awesome.<0x0A>It's also easier to see things in categories, here, than in knitting forums. I have the question on the back burner. I have the question that has been bugging me for ever. I have the question that I don't know 
 Inference time:    61.8550034      seconds
   2.05318880     tokens/second
 Timings
           1   118.843750    
           2  0.187500000    
           3   2.86718750    
           4   349.429688    
           5   14.7890625
```

## Notes 

Most of the model is well parallelized, I estimate it runs almost 10x faster on my 12 virtual core machine since I parallelized it. The compiler options are for maximum speed, and from my brief evaluation, gfortran is successfully using SIMD (AVX2 on my machine). In my tests it runs only very slightly slower than llama.cpp. I welcome advice on how to make it faster (on a CPU).

Originally I based this off of Andrej Karpathy's llama.c. Check the "legacy" branch for a bit more information. The model file format is the same one he used, and the legacy branch will run his "tiny" models. I broke the format here it use 16-bit quantization in the majority of the weights.

Confusingly, the four_bit branch uses the 32-bit llama.c model format. I plan to update it to use converted 4-bit ggml files directly.

As explained more below, I don't to make a complicated program with lots of options, I'd rather have separate versions that can be adapted for different things.


## Some extra information

I wrote [earlier](http://marble.onl/posts/why_host_your_own_llm.html) that I think language model *inferenece* should be self-hosted for most non-trivial uses. A big reason for this is that LLMs are still a new and rapidly evolving technology and that being able to "hack" the implementation is important to make the best use of them. A corollary to being able to hack the implementation is being able to easily understand and modify the code. The requirements for a hackable model are at odds with the requirements for a framework that has lots of composable parts and works across many platforms. What I want, and see a niche for, is something that's dead simple, where the only abstraction is linear algebra and matrix operations, but is also fast enough to run inference at competitive speeds on normal hardware. 

[Pytorch](https://pytorch.org/) is a full featured framework but is highly abstracted and not optimized for CPU inference. [Llama.cpp / ggml](https://github.com/ggerganov/llama.cpp) is well optimized for a wide range of hardware and has a simpler project structure compared to pytorch that increases hackability. However as of writing, ggml.c is 20k lines and llama.cpp is 7k. The hand optimization across many platforms plus big range of options (all of which make it a good, full featured software project) make it heavy to work with. [Llama2.c](https://github.com/karpathy/llama2.c) (the names are confusing and I may change the name of this project) is very hackable (although less than when it started) and simple to understand. It is not optimized; while in principle it could be, it will still be a C program that requires memory management and manual vector / matrix operations.

| | Pytorch | llama.cpp | llama2.c | llama2.f90 |
|-|---------|-----------|----------|------------|
|Good abstraction| x | x | | |
|Broad hardware support| x | x | | |
|Simple & Hackable| | | x | x |
|Fast| | x | | x |
|Memory and linalg| x | | | x |


What I want to do with Llama2.f90 is retain the hackability of llama2.c, but with the speed of Llama.cpp (currently we achieve comparable speeds on CPU) and the matrix and memory support of Fortran. So far optimization has not significantly diminished the readability or understandability of the code. The goal is not a framework that can be called from other programs, but example source code that can be modified directly for custom use. The hope is that such modifications will be as easy or easier than working with a high level framework. At the same time, we provide the capability of running an LLM from the command line. 

Additional options, such as quantization (under development), I prefer to include in dedicated programs instead of as branches of one main program. Likewise if we decide to support another model. In this way (hopefully) we keep everything simple and easy to use and hack elsewhere.


