from aslearn.kernel.kernels import RBF, White
import torch as th
device = "cuda" if th.cuda.is_available() else "cpu"

### individual test

print("--------individual test--------")
# RBF
print("--------RBF--------")
# use scalar situation to test the broadcasting of different input-output dimensions
for dim_in in range(1,100,5):
    for dim_out in range(1,100,5):
        for l in th.linspace(0.1,50,5):
            for c in th.linspace(0.1,50,5):
                kernel = RBF(dim_in=dim_in, dim_out=dim_out, l=l.item(), c=c.item())
                X = th.randn(10, dim_in).double().to(device)
                Y = th.randn(10, dim_in).double().to(device)
                ker_res = kernel(X, Y)
                X = X.view(10, dim_in, 1) # (N, nx, 1) # for normal use, for all output, they share a single data
                Y = Y.view(10, dim_in, 1) # (N, nx, 1)
                
                X = (X/l).permute(2,0,1) # shape: (1, N, nx)
                Y = (Y/l).permute(2,0,1) # shape: (1, M, nx), let's regard ny axis as batch
                distance = th.cdist(X, Y) # (1, N, M)
                hand_res = c * th.exp( - distance ** 2 )
                assert th.allclose(hand_res, ker_res)
                
print("ALL TESTS PAST")

# compound test
print("--------compound test--------")
kernel1 = RBF(dim_in=2,dim_out=1)
kernel2 = White(dim_in=2,dim_out=1)
kernel3 = RBF(dim_in=2,dim_out=1)
kernel_c1 = kernel1 * kernel3 + kernel2
kernel_c2 = kernel1 + kernel3 + kernel2
kernel_c3 = (kernel1 + kernel3) ** 2. + kernel2
kernel_c4 = kernel1 * kernel3 ** 2. + kernel2

X = th.randn(4,2).double().to(device)
res_ker1 = kernel1(X, X)
res_ker2 = kernel2(X, X)
res_ker3 = kernel3(X, X)

diag_matrix = th.eye(len(X)).unsqueeze(dim=0).to(device).double()
c1_res = kernel_c1(X, X) + diag_matrix * kernel_c1.noise(X, X)
c2_res = kernel_c2(X, X) + diag_matrix * kernel_c2.noise(X, X)
c3_res = kernel_c3(X, X) + diag_matrix * kernel_c3.noise(X, X)
c4_res = kernel_c4(X, X) + diag_matrix * kernel_c4.noise(X, X)

c1_gt = res_ker1 * res_ker3 + res_ker2
c2_gt = res_ker1 + res_ker3 + res_ker2
c3_gt = (res_ker1 + res_ker3)**2. + res_ker2
c4_gt = res_ker1 * res_ker3**2. + res_ker2


assert th.allclose(c1_res, c1_gt)
assert th.allclose(c2_res, c2_gt)
assert th.allclose(c3_res, c3_gt)
assert th.allclose(c4_res, c4_gt)
print("ALL TESTS PAST")

# white warning test
print("--------white warning test--------")
kernel1 * kernel2

# gradient test
print("--------gradient test--------")

X = th.randn(20,2).double().to(device)
kernel1 = RBF(dim_in=2,dim_out=1)
kernel2 = White(dim_in=2,dim_out=1)
kernel3 = RBF(dim_in=2,dim_out=1)
kernel_c3 = (kernel1 * kernel3)**2. + kernel2
res = kernel_c3(X, X) + kernel_c3.noise(X, X)
res = res.sum()
res.backward()

for name, param in kernel_c3.named_parameters():
    if 'exponent' not in name:
        assert param.grad != None, "This param hasn't been integrated in the computational graph"
print("ALL TESTS PAST")

print("--------ALL TESTS PAST--------")