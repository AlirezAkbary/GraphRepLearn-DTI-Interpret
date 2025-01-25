import numpy as np

def mse(y, f):
    return ((y - f) ** 2).mean(axis=0)

def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z += 1
                u = f[i] - f[j]
                if u > 0:
                    S += 1
                elif u == 0:
                    S += 0.5
            j -= 1
        i -= 1
        j = i - 1
    return S / z if z != 0 else 0.0
