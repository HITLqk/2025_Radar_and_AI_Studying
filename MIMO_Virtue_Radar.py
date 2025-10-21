import numpy as np
import matplotlib.pyplot as plt

# 参数设置
lambda_ = 1
d = lambda_ / 2
Mt, Mr = 4, 4

dt = np.arange(Mt) * d
dr = np.arange(Mr) * d
D_virtual = np.add.outer(dt, dr).flatten()

plt.stem(D_virtual, np.ones_like(D_virtual))
plt.title("MIMO虚拟阵列位置")
plt.xlabel("位置 (m)")
plt.ylabel("阵元存在")
plt.show()
