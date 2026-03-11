import numpy as np
from matplotlib import pyplot as plt
def normal_distribution(x, mu=0, sigma=1):
    coeff = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    exponent =  - (x - mu) ** 2 / (2 * sigma ** 2)
    return coeff * np.exp(exponent)





if __name__ == "__main__":
    