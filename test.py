import numpy as np
import torch


def permute(x, n_in, kernels_layer, num_channels_to_permute):
    x_new = torch.clone(x)
    for i in range(num_channels_to_permute):
        idx = torch.randperm(kernels_layer)
        idx = (idx * n_in) + i
        print(idx)
        print(np.arange(i, x.shape[1], n_in))
        x_new[:, i:x.shape[1]:n_in, :, :] = x[:, idx, :, :]
    return x_new

# 5 filters
x = torch.zeros(100, 20, 5, 5)

permute(x, 4, 5, 4)


# 0 1 2 3
# 4 5 6 7
# 8 9 10 11
# 12 13 14 15
# 16 17 18 19