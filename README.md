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
