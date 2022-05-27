
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch
from einops import rearrange
# # aq = torch.Tensor(3,6)
# # bq = rearrange(aq, "b (h d) -> b h d", h=2)
# # print(bq)

# # ak = torch.Tensor(5,6)
# # bk = rearrange(ak, "b (h d) -> b h d", h=2)
# # print(bk)
# # similarity = torch.einsum('qhd, khd -> qhk', bq, bk)
# # print(similarity)
# # print( similarity.mean(-1))


# x = torch.Tensor([[1,0,0,0,0,1,0,0],[1,0,0,0,0,0,1,0],
# [0,0,1,1,0,0,0,0], [1,1,0,0,0,0,0,0],[0,0,0,0,0,1,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,1,0,0],
# [0,1,0,0,0,0,0,0],[0,0,0,0,1,1,0,0],[0,0,1,0,0,1,0,0],
# [1,0,0,1,0,0,0,0],[0,0,0,1,1,0,0,1],
# [1,0,0,0,0,1,0,1]])
# eachimg_selected_box_nums = [2,5,3,2,1]

# boxes_to_gts_list = torch.Tensor([1,0,1,0,0,0,0,1,0,0,1,0,1])


# q = x
# k = x
# q = rearrange(q, "b (h d) -> b h d", h=2)
# k = rearrange(k, "b (h d) -> b h d", h=2)
# box_id = 0
# matrs = []
# matrs_for_loss = []
# querys = []
# keys = []
# att_drop = nn.Dropout(0)
# for boxNum in eachimg_selected_box_nums:
#     this_qs = q[box_id:box_id+boxNum]

#     # this_qs = F.softmax(this_qs, dim=-1)

#     querys.append(this_qs)
#     this_ks = k[box_id:box_id+boxNum]
#     # this_ks = F.softmax(this_ks, dim=-1)
#     keys.append(this_ks)
#     box_id+=boxNum
    

# # print(len(keys))#4
# for i, qs in enumerate(querys):
#     similarity_l = []
#     similarity_for_loss = []
#     for j, ks in enumerate(keys):
#         if i==j:
#             similarity = torch.einsum('qhd, khd -> qhk', qs, ks)
#             similarity_for_loss.append(rearrange(similarity, 'q h k -> k q h'))
     
#             continue
#         similarity = torch.einsum('qhd, khd -> qhk', qs, ks) #objq * h * objk
        
#         similarity_mean = similarity.mean(-1) 
#         similarity_max = similarity.max(-1)[0]
#         similarity_min = similarity.min(-1)[0]
#         similaritys = torch.stack([similarity_mean,similarity_max,similarity_min])
#         similarity_for_loss.append(rearrange(similarity, 'q h k -> k q h'))
#         similarity_l.append(similaritys) 
    
#     similarity_l = torch.stack(similarity_l) #imgk * 3 * objq * h

#     similarity_for_loss = torch.cat(similarity_for_loss) # (imgk * objk) * objq * h

    
#     avg_sim_matr = similarity_l.mean(0) # 3 * objq * h
#     max_sim_matr = similarity_l.max(0)[0]
#     min_sim_matr = similarity_l.min(0)[0]
#     matr = torch.stack([avg_sim_matr,max_sim_matr,min_sim_matr]) 
#     matr = rearrange(matr, 'n m o h -> o (n m h)', h = 2, n=3,m=3) # objq * (3*3*h)
#     similarity_for_loss = rearrange(similarity_for_loss, 'o n h -> n o h') #(imgk * objk) * objq * h ->  objq * (imgk * objk) * h
#     matrs.append(matr) # 
#     matrs_for_loss.append(similarity_for_loss)
    
# matrs = torch.cat(matrs) # all objs * (3*3*h)
# matrs_for_loss = torch.cat(matrs_for_loss) # (imgq * objq) * (imgk * objk) * h
# # matrs_for_loss = rearrange(matrs_for_loss, 'h n o -> n o h')
# print(matrs,matrs.shape)



# class MultiHeadCrossSimilarity(nn.Module):
#     def __init__(self, emb_size: int, num_heads: int = 8, dropout: float = 0):
#         super().__init__()
#         self.emb_size = emb_size
#         self.num_heads = num_heads
#         self.qemb = nn.Linear(emb_size, emb_size)
#         self.kemb = nn.Linear(emb_size, emb_size)
#         self.att_drop = nn.Dropout(dropout)
#         self.sigmod = nn.Sigmoid()

#     def forward(self, eachimg_selected_box_nums, x: Tensor) -> Tensor:
#         q = self.qemb(x)[0]
#         q = rearrange(q, "b (h d) -> b h d", h=2)
#         k = self.kemb(x)[0] 
#         k = rearrange(k, "b (h d) -> b h d", h=2)
#         max_ = q.size(0) - 1
#         box_id = 0
#         matrs = []
#         querys = []
#         keys = []
#         for boxNum in eachimg_selected_box_nums:
#             this_qs = q[box_id:box_id+boxNum]
#             querys.append(this_qs)
#             this_ks = k[box_id:box_id+boxNum]
#             keys.append(this_ks)
#             box_id+=boxNum

#         for i, qs in enumerate(querys):
#             similarity_l = []
#             for j, ks in enumerate(keys):
#                 if i==j:
#                     continue
#                 similarity = torch.einsum('qhd, khd -> qhk', qs, ks)
            
#                 similarity_mean = similarity.mean(-1) 
#                 similarity_max = similarity.max(-1)[0]
#                 similarity_min = similarity.min(-1)[0]
#                 similarity = torch.stack([similarity_mean,similarity_max,similarity_min])
                
#                 similarity_l.append(similarity) 
                
#             # print(len(similarity_l))

#             similarity_l = torch.stack(similarity_l) #imgk * 3 * objq * h
#             # print(similarity_l)
#             print('---------')
            
#             avg_sim_matr = similarity_l.mean(0) # 3 * objq * h
#             max_sim_matr = similarity_l.max(0)[0]
#             min_sim_matr = similarity_l.min(0)[0]
#             matr = torch.stack([avg_sim_matr,max_sim_matr,min_sim_matr]) 
#             matr = rearrange(matr, 'n m o h -> o (n m h)', h = 2, n=3,m=3) # objq * (3*3*h)
#             matrs.append(matr) # 
#             # print(matr)

#         matrs = torch.cat(matrs) # all objs * (3*3*h)

#         return matrs, q, k

# def prepare_for_infNCE(boxes_to_gts_list, q, k, heads=8):
#     loss_fun = InfoNCE()
#     losses = torch.Tensor(0, requires_grad=True)
#     q = rearrange(q, "b h d -> h b d", h=heads)
#     k = rearrange(k, "b h d -> h b d", h=heads)
#     for h in heads:
#         q_this_h = q[h]
#         k_this_h = k[h]

#         inds1 = [i for i,x in enumerate(boxes_to_gts_list) if x == 1]
#         inds0 = [i for i,x in enumerate(boxes_to_gts_list) if x == 0]

#         cos_q = q_this_h[inds1]
#         cos_k = k_this_h[inds1]
#         other_q = q_this_h[inds0]
#         other_k = k_this_h[inds0]

#         loss_co2other = loss_fun(cos_q, cos_k, other_k)
#         loss_other2co = loss_fun(other_q, other_k, cos_k,negative_mode='other2co')
#         losses = losses + loss_co2other + loss_other2co
#     return losses

# import torch
# import torch.nn.functional as F
# from torch import nn

# __all__ = ['InfoNCE', 'info_nce']


# class InfoNCE(nn.Module):
#     """
#     Calculates the InfoNCE loss for self-supervised learning.
#     This contrastive loss enforces the embeddings of similar (positive) samples to be close
#         and those of different (negative) samples to be distant.
#     A query embedding is compared with one positive key and with one or more negative keys.
#     References:
#         https://arxiv.org/abs/1807.03748v2
#         https://arxiv.org/abs/2010.05113
#     Args:
#         temperature: Logits are divided by temperature before calculating the cross entropy.
#         reduction: Reduction method applied to the output.
#             Value must be one of ['none', 'sum', 'mean'].
#             See torch.nn.functional.cross_entropy for more details about each option.
#         negative_mode: Determines how the (optional) negative_keys are handled.
#             Value must be one of ['paired', 'unpaired'].
#             If 'paired', then each query sample is paired with a number of negative keys.
#             Comparable to a triplet loss, but with multiple negatives per sample.
#             If 'unpaired', then the set of negative keys are all unrelated to any positive key.
#     Input shape:
#         query: (N, D) Tensor with query samples (e.g. embeddings of the input).
#         positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
#         negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
#             If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
#             If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
#             If None, then the negative keys for a sample are the positive keys for the other samples.
#     Returns:
#          Value of the InfoNCE Loss.
#      Examples:
#         >>> loss = InfoNCE()
#         >>> batch_size, num_negative, embedding_size = 32, 48, 128
#         >>> query = torch.randn(batch_size, embedding_size)
#         >>> positive_key = torch.randn(batch_size, embedding_size)
#         >>> negative_keys = torch.randn(num_negative, embedding_size)
#         >>> output = loss(query, positive_key, negative_keys)
#     """

#     def __init__(self, temperature=0.1, reduction='mean'):
#         super().__init__()
#         self.temperature = temperature
#         self.reduction = reduction

#     def forward(self, query, positive_key, negative_keys=None, negative_mode='co2other'):
#         return info_nce(query, positive_key, negative_keys,
#                         temperature=self.temperature,
#                         reduction=self.reduction,
#                         negative_mode=negative_mode)


# def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='co2other'):
#     # Check input dimensionality.
#     if query.dim() != 2:
#         raise ValueError('<query> must have 2 dimensions.')
#     if positive_key.dim() != 2:
#         raise ValueError('<positive_key> must have 2 dimensions.')
#     if negative_keys is not None:
#         if negative_mode == 'co2other' and negative_keys.dim() != 2:
#             raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'co2other'.")
   
#     # Check matching number of samples.
 
#     # Embedding vectors should have same number of components.
#     if query.shape[-1] != positive_key.shape[-1]:
#         raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
#     if negative_keys is not None:
#         if query.shape[-1] != negative_keys.shape[-1]:
#             raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

#     # Normalize to unit vectors
#     # query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
#     if negative_keys is not None:
#         # Explicit negative keys

#         # Cosine between positive pairs
#         if negative_mode == 'co2other':
#             positive_logit = torch.mean(query @ transpose(positive_key),dim=1, keepdim=True)
        

#         elif negative_mode == 'other2co':
#             positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)
        
#         negative_logits = query @ transpose(negative_keys)
           

#         # First index in last dimension are the positive samples
#         logits = torch.cat([positive_logit, negative_logits], dim=1)
#         print(logits)
#         labels = torch.zeros(len(logits), dtype=torch.long, device=query.device) # len = len of query
#         # print("2",logits)
#     else:
#         # Negative keys are implicitly off-diagonal positive keys.

#         # Cosine between all combinations
#         logits = query @ transpose(positive_key)

#         # Positive keys are the entries on the diagonal
#         labels = torch.arange(len(query), device=query.device)
#     return F.cross_entropy(logits / temperature, labels, reduction=reduction)

# def transpose(x):
#     return x.transpose(-2, -1)


# def normalize(*xs):
#     return [None if x is None else F.normalize(x, dim=-1) for x in xs]



# a = torch.Tensor([[1,0,0,0],[0,1,0,0]])
# b = torch.Tensor([[1,0,0,0],[0,1,0.1,0]])
# c = torch.Tensor([[0,0,1,0],[0,0,1,0],[0,0,1,0]])

# # a = torch.Tensor([[1,0,0,0],[1,0,0,0]])
# # b = torch.Tensor([[1,0,0,0],[1,0,0.1,0]])
# # c = torch.Tensor([[0,0,1,0],[0,0,1,0],[0,0,1,0]])

# positive_logit = torch.mean(a@b.transpose(-2,-1),dim=1, keepdim=True)

# negative_logits = a@c.transpose(-2,-1)
# # print(positive_logit)
# # print(negative_logits)
# # print(torch.arange(3))
# logits = torch.cat([positive_logit, negative_logits], dim=1)
# labels = torch.zeros(len(logits), dtype=torch.long)
# # print(logits)
# ls = F.cross_entropy(logits , labels)

# # print(F.softmax(logits, dim=-1))
# # print(logits, labels)

# # print(ls)

# loss = InfoNCE()

# query = a
# positive_key = b
# negative_keys = c
# # output = loss(query, positive_key, negative_keys)
# output = loss(query, positive_key, negative_keys,negative_mode='other2co')
# print(output)

# import os
# img_root = '../CoSOD/Data/DUTS_class/img/'
# classes = os.listdir(img_root)
# # print(classes)
# img_dir_paths = list(map(lambda x: os.path.join(img_root, x), classes))
# # print(img_dir_paths)
# for i in img_dir_paths:
#     print(os.listdir(i))
#     break
import numpy as np
a = torch.randn(1,3,8,8)
b = torch.zeros(1,1,8,8)
c = a*b
print(c.shape)
c = a*np.squeeze(b)
print(c.shape)
