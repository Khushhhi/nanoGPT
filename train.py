import torch 
import torch.nn as nn 
from torch.nn import functional as f 
torch.manual_seed(1337)

#tokenization - process of converting a sequence of strings into integers from the given vocabulary
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    # print("length of dataset in characters: ", len(text))
    # print(text[:1000])

    #unique characters in the text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # print(''.join(chars))
    # print(vocab_size)

    #tokenization - process of converting a sequence of strings into integers from the given vocabulary
    stoi = { ch:i for i,ch in enumerate(chars)} #stoi = string to integer
    itos = { i:ch for i,ch in enumerate(chars)}#itos = integer to string

    encode = lambda s: [stoi[c] for c in s] #encoder: takes a string, outputs a list of integers
    decode = lambda l: "".join([itos[i] for i in l]) #decoder: takes a list of integers, outputs a sring
    
    # print(encode("hii"))
    # print(decode(encode("hii")))

    #enocding the entire dataset
    data = torch.tensor(encode(text), dtype=torch.long)
    # print(data.shape, data.dtype)
    # print(data[:1000])

    #splitting the data into train and validation sets
    n = int(0.9*len(data)) #first 90% will be training data, rest val
    train_data = data[:n]
    val_data = data[n:]
    # print(train_data)
    # print(val_data)

    block_size = 8
    # print(train_data[:block_size+1])

    x = train_data[:block_size] #[18,47,56,57,58,1,15,47]
    y = train_data[1:block_size+1] #[47,56,57,58,1,15,47,58]
    for t in range(block_size):
        # print(t)
        context = x[:t+1] #18
        target = y[t] #47
        # print(f"when input is {context} the target: {target}")

    torch.manual_seed(1337)
    batch_size = 4 #how many independatn sequences will we process in parallel?
    
    def get_batch(split):
        #generatign a small batch of data of inputs x and targets y
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])

        return x,y

    xb, yb = get_batch('train')
    # print('inputs:')
    # print(xb.shape)
    # print(xb) #input to the tranformer
    # print("targets: ")
    # print(yb.shape)
    # print(yb)

    # print('-----')

    for b in range(batch_size): #batch dimension
        for t in range(block_size):
            context = xb[b, :t+1]
            target = yb[b,t]
            # print(context.tolist())
            # print(f"when input is {context.tolist()} the target is {target}")


#using the bigram neural network to feed the context data

