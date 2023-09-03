# llama2.f90
Toy LLaMA2 model inference in Fortran

## Notice

This is based on llama2.c by Andrej Karpathy https://github.com/karpathy/llama2.c It's based on a version of the code circa July 28, 2023. The project has since become way more complicated.

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

- 1 based indexing
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

Run
```bash
./llm 0.9 "There was a man"
There was a man and his wife. They lived in a small house. The man was bald and the wife was Tom. The man liked to drive his car.
One day, Tom and his wife went outside. They saw a big tree. Tom said, "I want to drive the tree!" He got in his car and started to drive around the tree.
Tom's wife watched him. She smiled. She said, "Tom, you can drive when I am with you." Tom smiled back. They played together until it was time to go home.
<s>
 Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine and pick flowers. One day, she saw a big thing in the sky. It was a plane! It made a loud noise, but it didn't hurt much.
Lily climbed down from the cloud and went to the beach. She saw a dolphin swimming in the water. The dolphin was so pretty, Lily wanted to play with it. But the dolphin was too far away and wouldn't come closer.
Lily was sad that she couldn't playful shore. She saw the dolphbled a dol 
```
