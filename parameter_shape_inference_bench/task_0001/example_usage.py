import numpy as np
from forward import forward

x = np.random.randn(10)
y = forward(x)
print(y)
