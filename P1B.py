import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.space_mission import SpaceMission
utils.check_for_newer_version()
#@jit(nopython = True) #speed(?)

L = 10e-6 #Bredde på boksen i meter
T = 3000 #Gassens temperatur i kelvin
N = 10000 #Antall partikler
t_c = 10e-9 #Tid
dt = 10e-12 #Tids intervall i s'
t = np.arange(0,t_c,dt) #Tidsarray definert vha Tid og tidssteg

def simulate_engine_performance(npb): #npb = number_of_particles_in_box. Code for 1 B and C
    rows = 2 #For vectors
    cols = npb

    #Bruh moment, brukte fusst vanlig velocity istedenfor vektorform
    #expected_velocity = 2*np.sqrt((2 * const.k_B * T) / (np.pi * const.m_H2) )#Formel fra A.1. Variabel: T
    #excpected_velocity_squared = 3 * const.k_B * T / const.m_H2               #Formel fra A.3. Variabel: T
    #scale = np.sqrt(excpected_velocity_squared - expected_velocity**2)   #<v^2> og <v>^2 fra over.

    pos = L * np.random.rand(rows, cols) #Particle positions
    loc = 0
    scale = np.sqrt(const.k_B * T / const.m_H2) #Må bruke for vektorer. Stden stod i boka.
    vel = np.random.normal(loc = loc, scale = scale, size=(rows, cols)) #loc = mean, scale = standard deviation(std)
    for m in range(len(t)): #tidssteg
        pos += dt * vel #Euler cromer 

        x1 = np.where(pos[0] >= L)[0] 
        x2 = np.where(pos[0] <= 0)[0] #Kollisjon i boksen
        y1 = np.where(pos[1] >= L)[0] 
        y2 = np.where(pos[1] <= 0)[0]
        vel[0][x1] = -vel[0][x1]
        vel[0][x2] = -vel[0][x2]
        vel[1][y1] = -vel[1][y1]
        vel[1][y2] = -vel[1][y2]
        print('lol')

    #return tbp #thrust per box
simulate_engine_performance(N)
plt.show()
