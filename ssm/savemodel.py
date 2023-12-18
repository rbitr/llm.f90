import sys
import struct
import json
import torch
import numpy as np

#from transformers import AutoModel, AutoTokenizer 
#from sentence_transformers import SentenceTransformer
import re

if len(sys.argv) > 1:
    dir_model = sys.argv[1]
else:
    dir_model = "."

#with open(dir_model + "/tokenizer.json", "r", encoding="utf-8") as f:
#    encoder = json.load(f)

with open(dir_model + "/config.json", "r", encoding="utf-8") as f:
    hparams = json.load(f)

#with open(dir_model + "/modules.json", "r", encoding="utf-8") as f:
#    modules = json.load(f)

list_vars =  torch.load(dir_model + "/pytorch_model.bin")


def strip(x: str):
    #x = "auto_model." + x
    print(x,end="")
    y = list_vars[x]
    assert y.view(-1)[0].dtype == torch.float32
    print(y.shape)
    return y.numpy()

if len(sys.argv) > 2:
    outfile = sys.argv[2]
else:
    outfile = dir_model + "/model_converted_full.bin"

UNUSED = 0
# the actual size is different than the config
real_vocab_size = list_vars['backbone.embedding.weight'].shape[0]
with open(outfile,mode='wb') as of:
        #write up front stuff
        header = struct.pack(
        'iiiiiii',
        hparams['d_model'], UNUSED, hparams['n_layer'], UNUSED, 
        UNUSED, real_vocab_size, UNUSED,
        ) 
        of.write(header)

        w = strip('backbone.embedding.weight')
        of.write(memoryview(w))

        layers = hparams['n_layer']

        for l in range(layers):
            w = strip(f'backbone.layers.{l}.mixer.D')
            of.write(memoryview(w))

        for l in range(layers):
            w = strip(f'backbone.layers.{l}.mixer.in_proj.weight')
            of.write(memoryview(w))

        for l in range(layers):
            w = strip(f'backbone.layers.{l}.mixer.conv1d.weight')
            of.write(memoryview(w))

        for l in range(layers):
            w = strip(f'backbone.layers.{l}.mixer.conv1d.bias')
            of.write(memoryview(w))

        for l in range(layers):
            w = strip(f'backbone.layers.{l}.mixer.x_proj.weight')
            of.write(memoryview(w))

        for l in range(layers):
            w = strip(f'backbone.layers.{l}.mixer.dt_proj.weight')
            of.write(memoryview(w))

        for l in range(layers):
            w = strip(f'backbone.layers.{l}.mixer.dt_proj.bias')
            of.write(memoryview(w))

        for l in range(layers):
            w = strip(f'backbone.layers.{l}.mixer.A_log')
            of.write(memoryview(w))

        for l in range(layers):
            w = strip(f'backbone.layers.{l}.mixer.out_proj.weight')
            of.write(memoryview(w))

        for l in range(layers):
            w = strip(f'backbone.layers.{l}.norm.weight')
            of.write(memoryview(w))

        w = strip('backbone.norm_f.weight')
        of.write(memoryview(w))

        w = strip('lm_head.weight')
        of.write(memoryview(w))

exit(0)

if len(sys.argv) > 3:
    vname = sys.argv[3]
else:
    vname = "tokenizer.bin"

vocab = encoder["model"]["vocab"]
# write out vocab
max_len = max([len(bytes(v,"utf-8")) for v in vocab])
print("Maximum word size: ", max_len)
with open(vname, "wb") as f:
    f.write(struct.pack("i", max_len))

    for v in vocab:
        vb = bytes(v,"utf-8")
        f.write(struct.pack("ii", 0, len(vb)))
        f.write(struct.pack(f"{len(vb)}s",vb))
