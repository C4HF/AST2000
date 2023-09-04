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
    Rnorm = np.sqrt(Rvec[0]**2 + Rvec[1]**2)
    epsilon0 = 8.854187817e-12
    E = (q1 / (4*np.pi*epsilon0)) * (Rvec/ Rnorm**3)
    return E

def visualize_field(q1,r1,q2,r2):
    r1 = np.array(r1) # Position of charges
    r2 = np.array(r2)
    a = 0.1e-3 # Size of lattice spacing
    N = 10 # Number of lattice elements in each direction
    rx = np.zeros((2*N+1,2*N+1),float)
    ry = rx.copy()
    Ex = rx.copy()
    Ey = rx.copy()
    for i in range(2*N+1):
        for j in range(2*N+1):
            rx[i][j] = a*(i-N)
            ry[i][j] = a*(j-N)
            r = np.array([rx[i][j], ry[i][j]])    
            E = efieldq(q1,r,r1) + efieldq(q2, r, r2)
            Ex[i][j] += E[0]
            Ey[i][j] += E[1]
    print(rx)
    print(Ex)
    plt.quiver(rx,ry,Ex,Ey)
    plt.scatter(r1[0], r1[1])
    plt.scatter(r2[0], r2[1])
    plt.axis('equal')
    plt.show()
q1 = -0.1e-6
q2 = 0.1e-6
r1 = [0,0]
r2 = [0,0.5e-3]
#visualize_field(q1,r1,q2,r2)
epsilon0 = 8.854187817e-12
q1 = 4e-9
q2 = -6e-9
F = q1*q2/(4*np.pi*epsilon0*5**(3/2))
print(-2*F,F)