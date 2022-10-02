import numpy as np
np.random.seed(6)
def polynomial_data(p=12, scale=3.5):
    X = np.linspace(-1.2,1.2,200)
    weight = np.random.uniform([-15]*p,[15]*p)
    Y = 0.
    for i in range(p):
        Y += weight[i] * X ** i
    Y += np.random.normal(scale=scale, size=[200])
    return X.reshape(-1,1), Y.reshape(-1,1)
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    X, Y = polynomial_data()
    plt.plot(X, Y, '.r')
    plt.show()