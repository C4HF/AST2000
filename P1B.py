import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.space_mission import SpaceMission
seed = utils.get_seed('axlhk')
from ast2000tools.solar_system import SolarSystem
system = SolarSystem(seed)
utils.check_for_newer_version()
#@jit(nopython = True) #speed(?)

#print('My system has a {:g} solar mass star with a radius of {:g} kilometers.'  
#      .format(system.star_mass, system.star_radius))

#for planet_idx in range(system.number_of_planets):  #Planet 0 er hjem planeten
 #   print('Planet {:d} is a {} planet with a semi-major axis of {:g} AU.'
  #        .format(planet_idx, system.types[planet_idx], system.semi_major_axes[planet_idx]))

#times, planet_positions = ... # Your own orbit simulation code
#system.generate_orbit_video(times, planet_positions, filename='orbit_video.xml')

L = 10e-6 #Bredde på boksen i meter
T = 3000 #Gassens temperatur i kelvin
N = 10000 #Antall partikler
t_c = 10e-9 #Tid
dt = 10e-12 #Tids intervall i s'
t = np.arange(0,t_c,dt) #Tidsarray definert vha Tid og tidssteg

def simulate_engine_performance(npb): #npb = number_of_particles_in_box. Code for 1 B and C
    a = 0   #Skal telle antall partikler som slipper ut
    rows = 3 #For vectors
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

        x_out = np.where(L/4 < pos[0] > (3/4)*L)[0]  #Ser etter partikler på vei ut utgangshullet 
        y_out = np.where(L/4 < pos[1] > (3/4)*L)[0]  #med areal 0.25L^2 midt på xy-planet
        #Funke ikkje sånn; Både x og y må være true
        
        for n in range(len(x_out)):     #Vil erstatta posisjonene som forsvinne ut med nye, uniformt fordelte posisjoner
            pos[0][x_out] =             #Og legga antall posisjoner ut i tellingen a

        x1 = np.where(pos[0] >= L)[0] 
        x2 = np.where(pos[0] <= 0)[0] #Ser etter kollisjoner i boksen
        y1 = np.where(pos[1] >= L)[0] 
        y2 = np.where(pos[1] <= 0)[0]
        z1 = np.where(pos[2] >= L)[0] 
        z2 = np.where(pos[2] <= 0)[0]

        vel[0][x1] = -vel[0][x1]
        vel[0][x2] = -vel[0][x2]
        vel[1][y1] = -vel[1][y1]
        vel[1][y2] = -vel[1][y2]
        vel[2][z1] = -vel[0][z1]
        vel[2][z2] = -vel[0][z2]
        


        plt.hist(pos)
        #Gjennomsnittlig energi per molekyl
        #Trykk
    #return tbp #thrust per box
x = simulate_engine_performance(N)
plt.show()
