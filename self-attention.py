import torch 
import torch.nn as nn 
from torch.nn import functional as F

# mathematical trick in self-attention inside a transformer

torch.manual_seed(1337)
B,T,C = 4,8,2 #batch, time, channel 
#information needs to be flown only from the previous context to make future predictiors. Future context should not be referred to.
x=torch.randn(B,T,C)
# print(x.shape)

'''
In every single batch, for every t token in that sequence, we want to calculate the average of all vectors in all the previous tokens and the current token. 
bagofwords (bow) = avg of tokens 
'''
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # (t,C)
        xbow[b,t] = torch.mean(xprev,0)
# print(x[0])
# print(xbow[0])      

#all of the above can be made much more efficient by using matrix multiplication

wei = torch.tril(torch.ones(T,T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (B,T,T) @ (B,T,C) ----> (B,T,C)
torch.allclose(xbow, xbow2)

# torch.manual_seed(42)
# a = torch.tril(torch.ones(3,3))
# a = a / torch.sum(a, 1, keepdim=True)
# b = torch.randint(0,10,(3,2)).float()
# c = a @ b # @ = multiplication of two matrices
# print('a=', a)
# print('b=', b)
# print('c=',c)