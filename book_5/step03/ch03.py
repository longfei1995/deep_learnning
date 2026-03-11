import numpy as np
import os
import matplotlib.pyplot as plt
def multivarite_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    D = len(x)
    norm_const = 1.0 / (np.power(2 * np.pi, D / 2) * np.power(det, 1.0 / 2))
    x_mu = x - mu
    result = np.exp(-0.5 * (x_mu @ inv @ x_mu.T))
    return norm_const * result


if __name__ == "__main__":
    # 加载数据
    path = os.path.join(os.path.dirname(__file__), "height_weight.txt")
    xs = np.loadtxt(path)

    # 计算均值和协方差矩阵
    mu = np.mean(xs, axis=0)
    cov = np.cov(xs, rowvar=False)
    
    