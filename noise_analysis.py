import numpy as np
from scipy import io


if __name__ == '__main__':
    data = io.loadmat('20121126NONESTILL.mat')['out0']
    ang = data[-10000:, 4:7]
    dev = ang - np.mean(ang, 0)
    std = np.sqrt(np.mean(dev ** 2))
    print('std =', std)
