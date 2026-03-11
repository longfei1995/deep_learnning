import os
import numpy as np
from matplotlib import pyplot as plt

path = os.path.join(os.path.dirname(__file__), "height.txt")

# normal distribution
def normal(x, mu=0, sigma=1):
    y = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return y


if __name__ == "__main__":
    xs = np.loadtxt(path)
    mu = np.mean(xs)
    sigma = np.std(xs)
    
    # sample
    sample = np.random.normal(mu, sigma, 10000)
    
    # plot histogram
    plt.hist(sample, bins='auto', density=True)
    plt.show()