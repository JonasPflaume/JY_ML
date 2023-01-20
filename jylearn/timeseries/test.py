import torch as th

A = th.randn(2,2,1)
B = th.randn(2,3,1)
res= th.einsum("bik,bjk->bij",A,B)
print(res)
print(A[0]@B[0].T)