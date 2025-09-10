

"""手搓transformer"""

import torch 
from torch import nn
import torch.nn.functional as F
import math 

"""搓Embedding"""

random_torch=torch.rand(4,4)

#将输出的词汇表索引转化为指定维度的Embedding

class TokenEmbedding(nn.Embedding):
    def __init__(self,vocab_size,d_model):
         super(TokenEmbedding,self).__init__(vocab_size,d_model,padding_idx=1)


class Positionalembedding(nn.Module):
     def __init__(self,d_model,max_len,device):
        super(Positionalembedding,self).__init__()
        self.encoding=torch.zeros(max_len,d_model,device=device)
        self.enconding.requires_grad=False
        pos=torch.arange(0,max_len,device=device)
        

