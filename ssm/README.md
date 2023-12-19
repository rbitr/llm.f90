# ssm.f90

<p align="center">
  <img src="assets/aaron.png" alt="Unleashing Mamba">
</p>

Fortran inference code for the [Mamba](https://github.com/state-spaces/mamba) state space model (ssm). Runs on CPU only for the moment but fast enough to use.

Disclaimer: this is a proof-of-concept implementation, there will be bugs and it still needs to be optimized for speed and cleaned up. The current model format will be migrated to gguf. 

Currently only tested on linux with gfortran. Please open an issue if you have trouble running it under other circumstances.

See the root [llm.f90](https://github.com/rbitr/llm.f90) readme for more information about the overall project.

## Getting started

### Clone the repo and build
```bash
git clone https://github.com/rbitr/llm.f90
cd llm.f90/ssm
make
```

### Get a model file


```bash
wget https://huggingface.co/SDFASDGA/llm/resolve/main/model-130m_converted_f32.bin 
```

This uses a converted format which is just all the weights packed together in a binary file plus a 28-byte header. Currently there are 130m and 790m converted files. You can also get the original models from the mamba authors and convert using `savemodel.py` included in this repo.

### Tokenizer

The mamaba models use a GPT NeoX 20B tokenizer. This repo includes a pre-converted version, `tokenizer.bin`. You can get the original here: https://huggingface.co/EleutherAI/gpt-neox-20b/tree/main and convert it with `convert_tokens.py`. 

### Run the model

```bash
$ ./llm -m model-130m_converted_f32.bin --ak -v -s tokenizer.bin -p "I stopped posting on knitting forums because" -n 100 -t 0.9
 Embedding dimension: (d_model)         768
 Layers:           24
 Vocabulary Size:        50280
 d_inner        1536
 dt_rank          48
 loaded embedding weights:    38615040
 loaded D weights:          24        1536
 loaded in projection weights:          24         768        3072
 loaded convolution weights:          24           4           1        1536
 loaded convolution bias:          24        1536
 loaded x_proj weights:          24        1536          80
 loaded delta projection weights:          24          48        1536
 loaded delta projection bias:          24        1536
 loaded A_log weights:          24          16        1536
 loaded out proj weights:          24        1536         768
 loaded mixer norm weights:          24         768
 loaded f norm weights:         768
 loaded classifer weights:         768       50280
 Loaded weights
          43
        6332
       16921
         328
       48143
       25279
         985
I stopped posting on knitting forums because they needed Warren to rework the 
story and he apparently stopped writing it because Warren was only blogging 
online and it wasn't working. I actually think the emails disconnected 
some people. They either had internet access or they weren't available.

Anonymous wrote:

Johnny and I did have conversations about what our names are, but you don't 
catch the whole story because it didn't stick, even though we were careful with 
what we said. I'm not 
 Inference time:    15.8719997      seconds
   6.23739910     tokens/second

```
