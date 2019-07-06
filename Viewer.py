import math
import time
import pickle
import numpy as xp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def Koordi(pom):
    return xp.copy(pom.reshape(len(pom), 7 * 3)[:, 0 : 3])

pickelPath = "C:\\Users\\sitnikov\\Documents\\Python Scripts\\data_.pickle"
PredelSumm = 1

f = open(pickelPath, "rb")
buffer = pickle.load(f)
f.close()

koordi = buffer['Varibles']['Chasichki'][0]
koordi = Koordi(koordi)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

x, y, z = xp.copy(koordi[:, :1]), xp.copy(koordi[:, 1:2]), xp.copy(koordi[:, 2:3])
ax.scatter(x, y, z, marker = 'o')

plt.show()
fig.savefig("test_rasterization1.svg", dpi=350)

fig, ax = plt.subplots()

s = xp.arange(0.0, len(buffer['Varibles']['Result']), 1)
ax.plot(s, buffer['Varibles']['Result'])

s = xp.arange(0.0, len(buffer['Varibles']['H']), 1)
ax.plot(s, buffer['Varibles']['H'])

plt.ylim(xp.min(buffer['Varibles']['H']), xp.max(buffer['Varibles']['H']))
ax.grid()
fig.savefig("test_rasterization2.svg")
plt.show()
