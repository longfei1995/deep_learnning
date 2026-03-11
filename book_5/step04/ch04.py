import os
import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
    # 加载数据
    path = os.path.join(os.path.dirname(__file__), "old_faithful.txt")
    xs = np.loadtxt(path)
    print(xs.shape)