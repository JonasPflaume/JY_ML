import numpy as np
from numba import njit
from numba.typed import List
import numba as nb

@njit("ListType(ListType((int64)))(int64[:], int64)")
def combinations(pool, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    n = len(pool)
    if r > n:
        print("Please set r < n !!!")
    indices = List()
    for i in range(r):
        indices.append(i)
    
    temp = List()
    for i in indices:
        temp.append(pool[i])
    res = List()
    res.append(temp)
    
    while True:
        for i in np.arange(r)[::-1]:
            if indices[i] != i + n - r:
                break
        else:
            return res
        indices[i] += 1
        for j in np.arange(i+1, r)[::-1]:
            indices[j] = indices[j-1] + 1
        temp = List()
        for i in indices:
            temp.append(pool[i])
        res.append(temp)