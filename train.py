with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    # print("length of dataset in characters: ", len(text))
    # print(text[:1000])
    #unique characters in the text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # print(''.join(chars))
    # print(vocab_size)

    stoi = { ch:i for i,ch in enumerate(chars)} #stoi = string to integer
    itos = { i:ch for i,ch in enumerate(chars)}#itos = integer to string

    encode = lambda s: [stoi[c] for c in s] #encoder: takes a string, outputs a list of integers
    decode = lambda l: [itos[i] for i in l] #decoder: takes a list of integers, outputs a sring
    