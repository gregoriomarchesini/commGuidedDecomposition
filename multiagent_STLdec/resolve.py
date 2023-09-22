import numpy as np

color = np.ones((1,4))
color[0,:3] = np.random.random(3)
print(color)
C = np.repeat(color, 10, axis=0)
print(C)
C[:,3] =C[:,3]*np.linspace(0.,1.,10)
print(C)
