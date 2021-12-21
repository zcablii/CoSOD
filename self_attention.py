import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
__all__ = ['MultiHeadAttention', 'ScaledDotProductAttention']

# class ScaledDotProductAttention(nn.Module):

#     def forward(self, query, key, value, mask=None):
#         print(query.shape,key.shape)
#         dk = query.size()[-1]
#         scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)
#         attention = F.softmax(scores, dim=-1)
#         return attention.matmul(value)

class ScaledDotProductAttention(nn.Module):

    def forward(self, query,q_k,q_v, key, value, mask=None):
        
        dk = query.size()[-1]
        q_att = query.matmul(q_k.transpose(-2, -1)) 
        q_att = torch.diagonal(q_att[0], 0).reshape((-1,1))
        # print(q_att)
        scores = query.matmul(key.transpose(-2, -1)) 
        # print(scores)
        scores = torch.cat((scores[0],q_att),-1).reshape(1,query.size()[1],-1)/ math.sqrt(dk+1)
        # print(scores)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        attention_ = attention[:,:,:-1]

        res = attention_.matmul(value)
        
        q_v_att = q_v*q_att
        res = q_v_att + res

        return res

class MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v, mask=None):
        
        q_k,q_v = self.linear_k(q), self.linear_v(q)
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        # print('q: ',q.shape)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q,q_k,q_v, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )


def randlist(random_int=10,num_range=(1, 100)):
    num_list = []
    r_min,r_max = num_range
    for count in range(random_int):
        num_list.append(random.randint(r_min,r_max))
    return num_list


# emb_len = 8
# img_num = 4


# objs = randlist(img_num,(3,9))
# obj_embs = []
# for obj in objs:
#     obj_embs.append(torch.randn(obj,emb_len))
# objs
# obj_embs_c = obj_embs.copy()

# # object on each img: [6, 6, 7, 4]
# # obj_embs[0].shape,obj_embs[1].shape
# # torch.cat((obj_embs[0],obj_embs[1]),1).shape

# att = MultiHeadAttention(emb_len,1)
# atts = []
# for i in range(len(objs)):

#     inds = i
#     print('PPPP ', obj_embs_c[inds])
#     q = torch.Tensor(obj_embs_c[inds].reshape(1,-1,8))
#     k = torch.Tensor([np.concatenate([x for i,x in enumerate(obj_embs_c) if i!=inds])]).reshape(1,-1,8)
#     print(q.shape,k.shape)
#     atts.append(att(q,k,k))
 


