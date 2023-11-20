# llm.f90 

(formerly llama2.f90)

Hackable large language model inference in pure Fortran. Builds to a ~100k executable that can be run efficiently on a CPU and has zero external dependencies. Between this and sibling project https://github.com/rbitr/ferrite you can create and customize a retrieval augmented (RAG) or other complete language model system.

## Getting started

The base implementation in the `master` branch runs on a single core only. See the roadmap below for more info on what has been done and what is planned.


### Clone the repo and build
```bash
git clone https://github.com/rbitr/llm.f90
cd llm.f90
make
```

### Get a model file (supports GGUF format)

This is a 1.1B parameter llama model converted into 16-bit gguf. See https://huggingface.co/Tensoic/Tiny-Llama-openhermes-1.1B-step-715k-1.5T for the model info

```bash
wget https://huggingface.co/SDFASDGA/llm/resolve/main/ggml-model-f16.gguf
```

### Run the model

```bash
$ ./llm -m ggml-model-f16.gguf -v -t 0.9 -p "I stopped posting in knitting forums because" -n 96
 GGUF Header Info
 Magic number:   1179993927
 Version:            3
 Tensor Count:                   201
 Key-Value Pairs:                    18
 general.architecture                                            
 llama                                                           
 general.name                                                    
 llm                                                             
 llama.context_length                                            
        2048
 llama.embedding_length                                          
        2048
 llama.block_count                                               
          22
 llama.feed_forward_length                                       
        5632
 llama.rope.dimension_count                                      
          64
 llama.attention.head_count                                      
          32
 llama.attention.head_count_kv                                   
           4
 llama.attention.layer_norm_rms_epsilon                          
   9.99999975E-06
 general.file_type                                               
           1
 tokenizer.ggml.model                                            
 llama                                                           
 tokenizer.ggml.tokens                                           
       32000
 tokenizer.ggml.scores                                           
       32000
 tokenizer.ggml.token_type                                       
       32000
 tokenizer.ggml.bos_token_id                                     
           1
 tokenizer.ggml.eos_token_id                                     
           2
 tokenizer.ggml.unknown_token_id                                 
           0
 Position      735611
 Deficit          26
 data offset      735617
 Embedding dimension:         2048
 Hidden dimension:         5632
 Layers:           22
 Heads:           32
 kv Heads:            4
 Vocabulary Size:        32000
 Sequence Length:         2048
 head size           64
 kv head Size          256
 loaded embedding weights:    65536000
 loaded rms att weights:       45056
 loaded wq weights:    92274688
 loaded wk weights:    11534336
 loaded wv weights:    11534336
 loaded wo weights:    92274688
 loaded ffn norm weights:       45056
 loaded w1 (gate) weights:   253755392
 loaded w2 (down) weights:   253755392
 loaded w3 (up) weights:   253755392
 loaded output norm weights:        2048
 loaded classifier weights:    65536000
 loading tokens
found 32000 tokens
found 32000 scores
 maximum token length           48
 Loaded weights
I stopped posting in knitting forums because they grew so toxic and self-centered. I have recommended listerine for every application to help with dryness and irritation and give a nice soothing rinsing effect. This is a common criticism from knitters with dry skin issues.<0x0A>I recently purchased some of Luxatox Extreme Dry Skin Balm (I almost called it Extreme Skin Care) and I love it! 
 Inference time:    14.2080011      seconds
   6.68637371     tokens/second
 Timings
           1   17.3333340    
           2   0.00000000    
           3   1.33333337    
           4   118.666664    
           5   12.0000000 
           5   17.3333340 
```

### Options

### Notes

The base version currently hard codes the model parameters. This is trivially changed with some uncommenting that will let you load any llama2 model. For anything much bigger (depending on your computer) the suggested branch is https://github.com/rbitr/llama2.f90/tree/version_0 than implements 16-bit floats and parallelism but has not been optimized. To use this branch you will have to get a .gguf version of the model and then convert it as described in the readme.

Models may load slightly faster if you convert to the "ak" file format (from Andrej Karpathy's llama2.c) and load that instead. 

## Features and Roadmap

If you want to use `llm.f90` for a project and need support, please get in touch. See the `motivation` section below for information about the "philosophy". We want any features added to not add complexity, so for example quantization will be written as a separate program.

- :white_check_mark: Speed: currently matches llama.cpp for single thread 32-bit operation (tested on a single intel machine so ymmv). This 16-bit branch runs at ~6.6/7.4 of the speed and is still under development
- :construction: Parallelism: see https://github.com/rbitr/llama2.f90/tree/version_0 (also with 16-bit quantization)
- :construction: Quantization: see https://github.com/rbitr/llama2.f90/tree/f16_convert and https://github.com/rbitr/llama2.f90/tree/four_bit_dev for 16-bit and 4-bit respectively
- :soon: Support for other models
- :soon: Test on other architectures machines (Apple, other ARM, etc). Please open issues for any feedback.
- :soon: ... 

Note that :construction: means features that have a "legacy" implementation that works but uses older model file formats and may have other breaking changes. The plan is to roll these into the current `master` branch while preserving speed optimizations and direct loading of gguf files.

## Motivation

See [here](http://marble.onl/posts/why_host_your_own_llm.html) for why language models *inferenece* should be self-hosted for most non-trivial uses. A big reason for this is that LLMs are still a new and rapidly evolving technology and that being able to "hack" the implementation is important to make the best use of them. A corollary to being able to hack the implementation is being able to easily understand and modify the code. The requirements for a hackable model are at odds with the requirements for a framework that has lots of composable parts and works across many platforms. There is a niche for, is something that's dead simple, where the only abstraction is linear algebra and matrix operations, but is also fast enough to run inference at competitive speeds on normal hardware. 

[Pytorch](https://pytorch.org/) is a full featured framework but is highly abstracted and not optimized for CPU inference. [Llama.cpp / ggml](https://github.com/ggerganov/llama.cpp) is well optimized for a wide range of hardware and has a simpler project structure compared to pytorch that increases hackability. However as of writing, ggml.c is 20k lines and llama.cpp is 7k. The hand optimization across many platforms plus big range of options (all of which make it a good, full featured software project) make it heavy to work with. [Llama2.c](https://github.com/karpathy/llama2.c) (the names are confusing and I may change the name of this project) is very hackable (although less than when it started) and simple to understand. It is not optimized; while in principle it could be, it will still be a C program that requires memory management and manual vector / matrix operations.

| | Pytorch | llama.cpp | llama2.c | llm.f90 |
|-|---------|-----------|----------|------------|
|Good abstraction| x | x | | |
|Broad hardware support| x | x | | |
|Simple & Hackable| | | x | x |
|Fast| | x | | x |
|Memory and linalg| x | | | x |


The plan is to retain the hackability of llama2.c, but with the speed of Llama.cpp (currently we achieve comparable speeds on CPU) and the matrix and memory support of Fortran. So far optimization has not significantly diminished the readability or understandability of the code. The goal is not a framework that can be called from other programs, but example source code that can be modified directly for custom use. The hope is that such modifications will be as easy or easier than working with a high level framework. At the same time, we provide the capability of running an LLM from the command line. 

Additional options, such as quantization (under development), are preferred to be added as in dedicated programs instead of as branches of one main program. Likewise if we decide to support another model. In this way (hopefully) we keep everything simple and easy to use and hack elsewhere.


