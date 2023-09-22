import numpy as np


x = np.random.random(20000)

C = np.array([[1., 0., 0.,1.]]*len(x[::100]))
print(C)
print(np.linspace(0.,1.,len(x[::100])))
print(C[:,3])
C[:,3] = np.linspace(0.,1.,len(x[::100]))
print(C[:,3])
    