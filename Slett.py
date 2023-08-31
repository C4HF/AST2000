import numpy as np
import matplotlib.pyplot as plt
def efieldq(q1,r,r1):
    # Input: charge q1 in Coulomb
    #        r: position to find field (in 1,2 or 3 dimensions) in meters
    #        r1: position of charge q1 in meters
    # Output: electric field E at position r in N/C
    r = np.array(r)
    r1 = np.array(r1)
    Rvec = r - r1
    Rnorm = np.sqrt(sum(r**2 + r1**2))
    epsilon0 = 8.854187817e-12
    E = Rvec * (q1 / (4*np.pi*epsilon0)) * (1/ Rnorm**3)
    return E
E = efieldq(1e-6, (0,0), (1,0))
print(E)