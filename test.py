#%%
import time
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import MathLib as m

xp = np

m.delta_T = 1
m.Hx_amplitud = 1
m.Hy_amplitud = 0
m.Frequency = .2
m.Radiuse = 3
m.Dlina_PAV = 1
m.GraniciVselennoy = 100

CisloChastic = 88

n = np.array([0.0,0.0,1.0])
mK = np.array([
    [0.0,0.0,5.0],
    [4.0,0.0,3.0],
    [3.0,4.0,0.0],
    [0.0,4.0,3.0],
    [4.0,0.0,-3.0],
    [-990.0,310.0,10.0]
])
mU = np.array([
    [0.0,0.0,-1.0],
    [0.0,-1.0,0.0],
    [-1.0,0.0,0.0],
    [0.0,0.0,1.0],
    [0.0,1.0,0.0],
    [1.0,0.0,0.0]
])
m1 = np.array([
    [0.0,0.0,-1.0],
    [0.0,-1.0,0.0],
    [-1.0,0.0,0.0],
    [0.0,0.0,1.0],
    [0.0,1.0,0.0],
    [1.0,0.0,0.0]
])
m2 = np.array([
    [0.0,0.0,-1.0],
    [0.0,-1.0,0.0],
    [-1.0,0.0,0.0],
    [0.0,0.0,1.0],
    [0.0,1.0,0.0],
    [1.0,0.0,0.0]
])
r = np.array([-90.0,110.0,10.0])
u = np.array([0.0,0.0,1.0])
# print(m.VneshPole(0, mU))
# print(m.SteerOttalk(mK,u,r))
# print(m.Moment(n,mK,mU,u,r))
# print(m.Sila(n,mK,mU,u,r))
# print(m.MathKernel(mK,mU,mK,mU,m1,m2,0, 0))
print(m.PorvrkaGrani(mK))

# koordi = MatrixUglSkorostiы

# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(111, projection='3d')

# x, y, z = xp.copy(koordi[:, :1]), xp.copy(koordi[:, 1:2]), xp.copy(koordi[:, 2:3])
# ax.scatter(x, y, z, marker = 'o')

# plt.show()