import numpy as np


if __name__ == "__main__":
    a = np.random.choice(4)
    b = np.random.choice(4, size=100, p=[0.1, 0.2, 0.3, 0.4])
    print(a, b)
    