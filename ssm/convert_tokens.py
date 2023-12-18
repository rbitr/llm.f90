# process GPTNeoX tokenizer model
# requires tokenizer.json in the current directory, get it using:
# overwrites any "tokenizer.bin" in the current directory
# wget https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/tokenizer.json

import json
import numpy as np
import struct

if __name__ == "__main__":

    with open("tokenizer.json") as f:
        t = json.load(f)

    tokens = t['model']['vocab']

    merged_tokens = [x.replace(" ", "") for x in t['model']['merges']]

    scores = [0]*len(t['model']['vocab'])
    max_score = len(scores)

    for id, tok in enumerate(merged_tokens):
        scores[tokens[tok]] = max_score - id

    vocab_list = [""] * len(scores)

    for k,v in tokens.items():
        vocab_list[v] = k

    extra = [x['content'] for x in t['added_tokens'][2:]]
    vocab_list.extend(extra)
    scores.extend([0]*len(extra))

    max_len = max([len(tok) for tok in vocab_list])

    vname = "tokenizer.bin"
    with open(vname, "wb") as f:
        f.write(struct.pack("i", max_len))

        for v,s in zip(vocab_list,scores):
            vc = v.replace("Ġ", " ").replace("Ċ","\n")
            vb = bytes(vc,"utf-8")
            f.write(struct.pack("f", np.float32(s)))
            f.write(struct.pack("i", len(vb)))
            f.write(struct.pack(f"{len(vb)}s",vb))




