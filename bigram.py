import torch 
import torch.nn as nn 
from torch.nn import functional as F
torch.manual_seed(1337)


#hyperparameters
batch_size = 32 #how many independatn sequences will we process in parallel
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else 'cpu' #if a gpu is used
eval_iters = 200
# ----------------

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


    # print(train_data[:block_size+1])

    x = train_data[:block_size] #[18,47,56,57,58,1,15,47]
    y = train_data[1:block_size+1] #[47,56,57,58,1,15,47,58]
    for t in range(block_size):
        # print(t)
        context = x[:t+1] #18
        target = y[t] #47
        # print(f"when input is {context} the target: {target}")


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

    @torch.no_grad() #tells pytorch that we dont need to do backprop here
    def estimate_loss():
        out={}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeroes(eval_iters)
            for k in range(eval_iters):
                X,Y = get_batch(split)
                logits, loss = model(X,Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
            model.train()
            return out
#using the bigram neural network to feed the context data
class BigramLanguageModel(nn.Module):
    
 
        '''
        each token directly reads off the logits for the next token from a lookup table
        logits = scores for the next character in sequence

        token_embedding table = vocab * vocab (in this case 65*65)
        when input is passed, every single input will refer to the embedding table and pluck out the corresponding row of the index. 
        Pytorch arranges all of this in a batch (4) by time (8) by channel tensor (OR VOCAB SIZE = 65)(B,T,C)
        
        Predicting what comes next with the invidividual identity of a single token.
        '''
        def __init__(self, vocab_size):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

        def forward(self, idx, targets=None):
             
            #idx and targets are both (B,T) tensor of integers
            logits = self.token_embedding_table(idx) # (B,T,C)

            if targets is None:
                loss = None
            else:
                B,T,C = logits.shape
                logits = logits.view(B*T,C) #stretching out the array so that it's two dimensional and C can be the second dimension

                #we need targets to be one-dimensional
                targets = targets.view(B*T)
                # loss function for evaluation
                loss = F.cross_entropy(logits, targets)
            return logits, loss
        
        #done determining the quality of the model

        #now generating the model 
        def generate(self, idx, max_new_tokens): #idx = current context of characters in some batch
             #idx is (B,T) array of indices in the current context
            for _ in range(max_new_tokens):
                #get the predictions
                logits, loss = self(idx) #loss will be ignored
                #focus only on the last time step
                logits = logits[:,-1,:] #becomes (B,C)
                #apply softmax to get probabilities 
                probs = F.softmax(logits, dim=-1) #(B,C)
                # sample from the distribution 
                idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
                # append sampled index to the running sequence 
                idx = torch.cat((idx,idx_next), dim=1) # (B, T+1)
            return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)
# print(logits.shape) #won't run with logits as (B,T,C) because PyTorch expects Channel (C) as second dimension so a (B,C,T)
# print(loss)

# idx = 
# print(decode(m.generate(idx=torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

#now training the model by creating a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    #sample a batch of data
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# print(loss.item())

#generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(idx=torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))





        