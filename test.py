#%%
import time
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, './lib')
import MathLib as m

xp = np

m.delta_T = 1
m.Hx_amplitud = 1
m.Hy_amplitud = 0
m.Frequency = .2
m.Radiuse = 3
m.Dlina_PAV = 1

a = []
for x in range(1000000):
    a.append([0, 7, 0,])
    a.append([0, 7.9, 0,])
    a.append([0, 10, 0,])

a.append([np.Infinity, np.Infinity, np.Infinity,])

a = xp.array(a, dtype = xp.float64)

b = xp.array([0, 0, 0,])
s = xp.array([1, 0, 0,])

t = time.time()
print(m.SteerOttalk(a, s, b))
print(time.time() - t)