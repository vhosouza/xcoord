import numpy as np

# %%
np.random.seed(10)
a = np.random.rand(6, 3)
a_d = np.diff(a, axis=0, append=a[np.newaxis, -2, :])
a_d[-1, :] *= -1
a_n = np.linalg.norm(a_d, axis=1)
a_f = np.absolute((a_d / np.linalg.norm(a_d, axis=1)[:, None]))
# a_f = a_d[:, :]/a_n[:]
# a_f_n = np.linalg.norm(a_f, axis=1)

# %%
a_d_loop = []
a_d_loop_n = []
for j in range(a.shape[0] - 1):
    d_loop = a[j + 1, :] - a[j, :]
    a_d_loop.append(d_loop)
    a_d_loop_n.append(np.linalg.norm(d_loop))