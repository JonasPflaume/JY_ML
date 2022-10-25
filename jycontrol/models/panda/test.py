from C_matrix import C_panda
from G_vector import G_panda
from M_matrix import M_panda
from param import get_param_CMA
import numpy as np
from numpy import cos, sin

param = get_param_CMA()
q = np.ones((7,))
dq = q.copy()
M = M_panda(param, q, cos, sin)

import time

s = time.time()

for _ in range(1000):
    M = M_panda(param, q, cos, sin)
    C = C_panda(param, q, dq, cos, sin)
    G = G_panda(param, q, cos, sin)

e = time.time()
print("Time cost: %.3f" % (e-s) )