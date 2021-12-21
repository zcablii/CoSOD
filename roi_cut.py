
import torch
import math
import torch.nn as nn
# pos_e = torch.rand(4, 10)
# for ind, i in enumerate (pos_e):
#     pos_e[ind][ind] = 1
# pos_e.reqires_grad = True
a = torch.Tensor([float('nan')]).cuda()
print(a[0])
print(torch.isnan(a[0]))
if torch.isnan(a[0]):
    print('sdf')

# a = torch.Tensor([[float('nan'),float('nan')]])
# print(math.isnan(a[0][0]))

# a = torch.Tensor([1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
#         1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1]) 
# b = torch.Tensor([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0,
#         0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1]) 
# c = torch.Tensor([0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0.,
#         1., 1., 1., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1.,
#         1., 1., 0., 1., 0., 0., 1., 1.]) 
# d = torch.Tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
#         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
#         0., 1., 0., 0., 0., 0., 1., 1.])

# print(len(a),len(b),len(c),len(d))
# print(a-b)
# for i in range(len(c)):
#     x = a-b
#     if x[i]==0 and c[i] ==1:
#         continue
#     elif x[i]!=0 and c[i] == 0:
#         continue
#     else:
#         print('eee')

# for i in range(len(c)):
#     if c[i]*a[i]==0 and d[i]==1:
#         print('eee')
#     elif c[i]*a[i]==1 and d[i]==0:
#         print('eee')

# a = [[0,0,1,1,1],[0,0,1,1,1],[0,0,1,1,1],[0,0,1,1,1],[0,0,1,1,1]]