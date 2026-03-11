import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    xs = np.arange(-2, 2, 0.1)
    ys = np.arange(-2, 2, 0.1)
    X, Y = np.meshgrid(xs, ys)
    Z = X ** 2 + Y ** 2
    
    
    # 等高线
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contour(X, Y, Z)
    plt.show()
