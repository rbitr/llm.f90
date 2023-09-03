# llama2.f90
Toy LLaMA2 model inference in Fortran

## Notice

This is based on llama2.c by Andrej Karpathy https://github.com/karpathy/llama2.c It's based on a version of the code circa July 28, 2023. The project has since become way more complicated.

## What

LLaMA2 type LLM inference architecture implemented as a single Fortran file. The toy model uses the "Tiny Stories" checkpoint from the llama2.c project, and reproduces those results.


## Why

I saw a discussion last week about Fotran and realized I had never used it and it looked interesting: https://news.ycombinator.com/item?id=37295144

This seemed like a good way to learn and explore Fortran for machine learning type programming

## Impressions

Things I liked:

- memory management
- array slicing
- intrinsics (higher level functions like `matmul`)
- seemingly fast (in my preliminary inaccurate test it beats llama2.c with default compiler options)
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
- parallelization (still not explored)

Overall, I can see how Fortran is competitive with C for compiled ML applications that have lots of linear algebra and run naively on the cpu. It has the right intrinsic functions, handles memory more easily than C (though declarations are awkward) and handles arrays natively. It remains to be seen how easily it works with other accelerators (BLAS, etc). 

## How

Clone the repo
```bash
git clone https://github.com/rbitr/llama2.f90
```

Download the trained model from HF
```bash
cd llama2.f90
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
``` 

Compile (only tested with GNU Fortran on Ubuntu)
```bash
gfortran llama2.f90 -o llm
```

Run (arguments for temperature and prompt are optinoal but are positional so if you specify a prompt you also need a temperature. T=0 is deterministic.
```bash
./llm 0.9 "There was a man"
There was a man who had lots of cheese. He wanted to keep some for himself, so he tried to cut it himself. But the cheese was too fragile and he cut it anyway. 
The man was very unhappy and he felt really bad. He realized that he should have left the cheese in the house. 
So the man decided to go to get it back, but were too late. Little didn't make it. 
The man never got his cheese back. He sadly ate it alone and never found another one.
<s>
 Once upon a time, there was an old dog named Buddy. Buddy loved to play outside in the dirty mud. One day, he found a big pile of corn. He thought it was fun to play with the corn.
Buddy's friend, a little girl named Lily, came by and saw the corn. She told Buddy not to play with it. But some of the corn was dirty. Buddy did not care. He just wanted to play with the corn. So, Buddy jumped into the mud and got himself all dirty.
Lily helped Buddy get clean. She told him it was okay to play with the
```
