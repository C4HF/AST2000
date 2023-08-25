import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.space_mission import SpaceMission
utils.check_for_newer_version()
#@jit(nopython = True) #speed(?)

L = 10e-6 #Bredde på boksen i meter
T = 3000 #Gassens temperatur i kelvin
N = 100 #Antall partikler
t_c = 10e-9 #Tid
dt = 10e-12 #Tids intervall i s'
t = np.arange(0,t_c,dt) #Tidsarray definert vha Tid og tidssteg

def simulate_engine_performance(npb): #npb = number_of_particles_in_box. Code for 1 B and C
    rows = npb #For vectors
    cols = 2

    #Bruh moment, brukte fusst vanlig velocity istedenfor vektorform
    #expected_velocity = 2*np.sqrt((2 * const.k_B * T) / (np.pi * const.m_H2) )#Formel fra A.1. Variabel: T
    #excpected_velocity_squared = 3 * const.k_B * T / const.m_H2               #Formel fra A.3. Variabel: T
    #scale = np.sqrt(excpected_velocity_squared - expected_velocity**2)   #<v^2> og <v>^2 fra over.

    pos = L * np.random.rand(rows, cols) #Particle positions
    loc = 0
    scale = np.sqrt(const.k_B * T / const.m_H2) #Må bruke for vektorer. Stden stod i boka.
    vel = np.random.normal(loc = loc, scale = scale, size=(rows, cols)) #loc = mean, scale = standard deviation(std)
    for m in range(10): #tidssteg
        for i in range(N): #Hvert molekyl ett tidssteg frem 
            pos[i][0] = np.array(pos[i][0] + (dt * vel[i][0]))  #Euler cromer x
            pos[i][1] = np.array(pos[i][1] + (dt * vel[i][1]))  #Euler cromer y
            if pos[i][0] >= L:          #Kollisjon i boksen
                vel[i][0] = -vel[i][0]
            if pos[i][0] <= 0:
                vel[i][0] = -vel[i][0]
            if pos[i][1] >= L:
                vel[i][1] = -vel[i][1]
            if pos[i][1] <= 0:
                vel[i][1] = -vel[i][1]
            plt.scatter(pos[i][0], pos[i][1])
    #return tbp #thrust per box
simulate_engine_performance(N)
plt.show()
