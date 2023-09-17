# llama2.f90
LLaMA2 model inference in Fortran

## Notice

This is based on llama2.c by Andrej Karpathy https://github.com/karpathy/llama2.c It was originally based on a version of the code circa July 28, 2023 and in partcular uses the model formats from that project.

If you have trouble running anything or have comments, suggestions, feature requests, etc, please get in touch or open an issue.

## What

LLaMA2 type LLM inference architecture implemented as a single Fortran file. Runs the "toy" models from llama2.c as well as the bigger LLaMA models.

For Fortran implementations of inference for various GPT models see https://github.com/certik/fastgpt 

Progress so far:

- runs LLaMA style toy models and base models (tested up to 7B)
- original implementation matches llama2.c for speed: see https://github.com/rbitr/llama2.f90/issues/3#issuecomment-1711905524 It should be faster now as more has been parallelized
- Now uses F16 quantization by default for most layers. Runs a 3B model a ~~0.1~~ ~~0.23~~ 0.8 Tokens/s on my 2021 Thinkpad in < 8GB RAM. "Fast"(er) handling of quantization through parallelization and lookup.
- Speed improvements over original implementation from parallelizing much of the inference process (by grouping operations and then parallelizing with OpenMP)

## Motivation and niche

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


## How

Clone the repo
```bash
git clone https://github.com/rbitr/llama2.f90
# also requires https://github.com/Maratyszcza/FP16/ for FP16
# otherwise use the old_master branch
```

Download the trained model from HF
```bash
cd llama2.f90
# 15M parameter model
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
# 42M parameter model
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
# 110M parameter model 
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
``` 

Compile (only tested with GNU Fortran on Ubuntu)
```bash
# requires the FP16 code referenced above 
# edit to select your appropriate compiler
make llm
```

Now supports some proper command line arguments

List with the following hack:
```bash
cat llama2.f90 | grep -A 1 "case ('" | awk '{$1=$1};1'
case ('-m', '--model')
! path to model file
--
case ('-p', '--prompt')
! prompt string
--
case ('-s', '--tokenizer')
! path to custom tokenizer
--
case ('-t', '--temperature')
! temperature scaling
--
case ('-n', '--num_tokens')
! number of tokens to generate, including prompt
--
case ('-v', '--verbose')
! print additional information
```

Run the model (see the bottom for the latest example):


```bash
./llm -m stories42M.bin -n 256 -t 0.9 -p "There was a woman that lived in a shoe"
There was a woman that lived in a shoe. She had lots of money, but she was very selfish.
One day, she went to the little girl who lived in the village. "Can you play with me?" she asked. 
The little girl was excited, but she was also scared. "What if the others show you how to be kind and generous?" she said.
The woman thought for a moment. "Well," she said, "I'll go play, and when I do, I'll be kind."
The little girl understood, and said goodbye. But she still watched on, ready to give the woman something, when she saw
<s>
 Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, Lily and her friend Timmy went on a walk in the forest. They saw many trees and birds, and it was very peaceful.
Suddenly, they heard a loud noise. A big bear appeared! "Quick, hurry!" cried Lily to Timmy. They ran as fast as they could, but the bear was getting closer.
Just when they thought they were going to get 
 Inference time:    3.83200002      seconds
   66.8058472     tokens/second
```

## LLaMA models

[llama2.c](https://github.com/karpathy/llama2.c/) comes with a utility `extract.py` that converts HF or LLaMA formatted models into the binary weights format currently used by llama2.f90.

__Open source model__

If you don't want to deal with Meta's license games, there are open source LLaMA models. I tried the Open LLaMA 3B model which is also good for experimenting because it's smaller.

```bash
$ # clone the HF repository
$ git clone https://huggingface.co/openlm-research/open_llama_3b_v2
$ python export.py open_llama_3b_v2_ak.bin --hf ./<path>/open_llama_3b_v2/ --version 0 
```
llama2.c also has a `tokenizer.py` file that can create a custom `tokenizer.bin` file from the `tokenizer.model` that comes with Open LLaMA. It saves the `tokenizer.bin` file in the model directory.

```bash
python tokenizer.py -t ./<path>/open_llama_3b_v2/
```

Then run manually specifying the tokenizer:

```bash
$ ./llm -v -m ../llama2.c/open_llama_3b_v2_ak.bin -s ../llama/open_llama_3b_v2/tokenizer.bin -p "On my day off I like to" -t 0.9 -n 128
 Embedding dimension:         3200
 Hidden dimension:         8640
 Layers:           26
 Heads:           32
 kv Heads:           32
 Vocabulary Size:       -32000
 Sequence Length:         2048
On my day off I like to relax and just do whatever I want. A big part of that is shopping for DIY projects. I love going through the pages of houses and decorating magazines, and figuring out what I like.<0x0A>The other part is searching the web for free ideas and tutorials anywhere I can find them. Then the day off is here and I have plenty of time to research, and finally the day off, and I can get to work.<0x0A>This DIY Palm Leaf Platter from Black Eiffel was one of those DIYs that I loved to look at, but I couldn’t justify going out and spending $ 
 Inference time:    252.324768      seconds
  0.507282734     tokens/second
```

__Meta's LLaMA2__

The current `extract.py` uses a lot of memory and could not covert the LLaMA 7B model in 32GB of RAM. Following [this](https://github.com/karpathy/llama2.c/issues/341) issue, you can download an older version that uses less memory at https://github.com/karpathy/llama2.c/blob/de005474d37d0cde1356739b8c79ebe7b42b5973/export_meta_llama_bin.py 

Once you jump through Meta's [hoops](https://github.com/facebookresearch/llama) to get the model, you can convert with

```bash
$ python export_meta_llama_bin.py <path_to_llama2_directory> llama2_7b_chat_ak.bin
```

In this case I used the chat version. It then runs as usual. On the machine I used (an Amazon EC2 with 8 cores and 32 GB RAM, don't ask), it runs very slowly (.25 tok/sec). More optimization is underway. Below is the output from a run of the 7B Chat model with an empty prompt. Inexplicably it begins in German and then switches into an english discussion of the difference between "Kraft" and "Kraftwerk". 

```bash
$ ./llm -v -m ./<path>/llama2_7b_chat_ak.bin 
 Embedding dimension:         4096
 Hidden dimension:        11008
 Layers:           32
 Heads:           32
 kv Heads:           32
 Vocabulary Size:       -32000
 Sequence Length:         2048
 Unterscheidung zwischen "Kraft" und "Kraftwerk"

"Kraft" und "Kraftwerk" sind two different German words that are often confused with each other. Here's a brief explanation of each word and how they differ:

1. "Kraft" (Strength, Power)

"Kraft" is a noun that means strength or power. It can refer to physical strength, mental strength, or the power of something like a machine or a natural force. For example:

* "Sie haben eine enorme Kraft" (You have a tremendous strength)
* "Die Kraft der Natur" (The power of nature)
2. "Kraftwerk" (Power Plant)

"Kraftwerk" is a noun that refers to a power plant or a facility that generates electricity. It can also refer to a company or organization that produces or distributes electricity. For example:

* "Das Kraftwerk in der Nachbarstadt produziert Strom für tausende Haushalte" (The power plant in the neighboring town produces electricity for thousands of households)
* "Die Kraftwerk AG ist ein wichtiger Energieversor 
 Inference time:    1056.36633      seconds
  0.242340162     tokens/second
```

## Quantized models

The original model used 32-bit reals and e.g needs almost 30GB to run the 7B model and will not run the 3B on my machine with 16GB RAM. The [sixteen_bit](https://github.com/rbitr/llama2.f90/tree/sixteen_bit) branch, has been merged into master and uses an [external](https://github.com/Maratyszcza/FP16/) C library to convert reals to 16-bit floats packed in Fortran `integer(2)`s. 

```bash
$ ./llm -m ./models/open_llama_3b_v2_ak.bin -s ./models/tokenizer_open3b.bin -t 0.9 -v -n 64 -p "I stopped posting in knitting forums because"
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
I stopped posting in knitting forums because my babes were too young to appreciate my hobby. But until the last few months, I'd managed to avoid the Instant Pot threads. Why? Because I hadn't quite made the leap from the not-on-Cooking-Foibles-people-ra 
 Inference time:    75.8880005      seconds
  0.830170751     tokens/second
```
