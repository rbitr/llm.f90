# llama2.f90
LLaMA2 model inference in Fortran

## Notice

This is based on llama2.c by Andrej Karpathy https://github.com/karpathy/llama2.c It's based on a version of the code circa July 28, 2023. The project has since become way more complicated.

## What

LLaMA2 type LLM inference architecture implemented as a single Fortran file. The toy model uses the "Tiny Stories" checkpoint from the llama2.c project, and reproduces those results. It's a bit of a wall of code, but should make sense in the context of the ipython notebook linked below and the llama2.c project. 

I've since found out there are Fortran implementations of inference for various GPT models: https://github.com/certik/fastgpt 

## Why

I saw a discussion last week about Fotran and realized I had never used it and it looked interesting: https://news.ycombinator.com/item?id=37295144

This seemed like a good way to learn and explore Fortran for machine learning type programming. 

See also https://github.com/rbitr/llama2.ipynb my implementation in python. Python was easier, Fortran was more fun!

## Impressions

Things I liked:

- memory management
- array slicing
- intrinsics (higher level functions like `matmul`)
- seemingly fast - with default compiler options, I get ~165 tokens/s with the 15M model from this implementation vs. 75 tokens/s from llama2.c on my 2021 Thinkpad. However, using OMP parallelization I get 250 tokens/s with llama2.c. 
- fun

Interesting things (I could be wrong about some):

- 1 based indexing (and off-by one errors can have a very subtle impact on output)
- column major arrays
- fixed size strings in arrays
- no native way to tell if a string has trailing spaces (important for tokenization)
- all the variables declared at the top
- trims matrix operation results to fit arrays? 
- harder to find answers to questions
- confusion about different standards; I think this is 2003 compliant
- parallelization  

Overall, I can see how Fortran is competitive with C for compiled ML applications that have lots of linear algebra and run naively on the cpu. It has the right intrinsic functions, handles memory more easily than C (though declarations are awkward) and handles arrays natively. It remains to be seen how easily it works with other accelerators (BLAS, etc). 

## How

Clone the repo
```bash
git clone https://github.com/rbitr/llama2.f90
```

Download the trained model from HF
```bash
cd llama2.f90
# 15M parameter model
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
# 42M parameter model
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
``` 

Compile (only tested with GNU Fortran on Ubuntu)
```bash
gfortran llama2.f90 -o llm
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
case ('-t', '--temperature')
! temperature scaling
--
case ('-n', '--num_tokens')
! number of tokens to generate, including prompt
--
case ('-v', '--verbose')
! print additional information
```

Run the model:


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

The current `extract.py` uses a lot of memory and could not covert the LLaMA 7B model in 32GB of RAM. Following [this](https://github.com/karpathy/llama2.c/issues/341) issue, you can download an older version that uses less memory at https://github.com/karpathy/llama2.c/blob/de005474d37d0cde1356739b8c79ebe7b42b5973/export_meta_llama_bin.py 

Once you jump through Meta's hoops to get the model, you can convert with

```bash
python export_meta_llama_bin.py <path_to_llama2_directory> llama2_7b_chat_ak.bin
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
