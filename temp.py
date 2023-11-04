with open('input.txt') as f:
    text = f.read()

#print('text length', len(text))
#print('first 1000: \n', text[:1000])

## here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

#print(''.join(chars))
#print(vocab_size)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#test_str = encode("hello there")
#print(test_str)
#print(decode(test_str))

import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
