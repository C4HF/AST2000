import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.space_mission import SpaceMission

seed = 57063
from ast2000tools.solar_system import SolarSystem

system = SolarSystem(seed)
utils.check_for_newer_version()
# @jit(nopython = True) #Optimalisering(?)
from mpl_toolkits import mplot3d  # Plotting

L = 10e-6  # Bredde på boksen i meter
T = 3000  # Gassens temperatur i kelvin
N = 100  # Antall partikler
t_c = 10e-9  # Tid
dt = 10e-12  # Tids intervall i s'
t = np.arange(0, t_c, dt)  # Tidsarray definert vha Tid og tidssteg
L = 10e-6  # Bredde på boksen i meter

fig = plt.figure()
ax = plt.axes(projection="3d")  # For 3d plotting av rakettmotoren
ax.plot3D([0, L], [0, 0], [0, 0], "green")  # Lager en yttre firkant på xy-planet
ax.plot3D([L, L], [0, L], [0, 0], "green")
ax.plot3D([0, 0], [0, L], [0, 0], "green")
ax.plot3D([0, L], [L, L], [0, 0], "green")

ax.plot3D(
    [0.25 * L, 0.75 * L], [0.25 * L, 0.25 * L], [0, 0], "green"
)  # Lager en indre firkant på xy-planet (utgangshull)
ax.plot3D([0.25 * L, 0.75 * L], [0.75 * L, 0.75 * L], [0, 0], "green")
ax.plot3D([0.25 * L, 0.25 * L], [0.25 * L, 0.75 * L], [0, 0], "green")
ax.plot3D([0.75 * L, 0.75 * L], [0.25 * L, 0.75 * L], [0, 0], "green")

# print('My system has a {:g} solar mass star with a radius of {:g} kilometers.'
#      .format(system.star_mass, system.star_radius))

# for planet_idx in range(system.number_of_planets):  #Planet 0 er )hjem planeten
#   print('Planet {:d} is a {} planet with a semi-major axis of {:g} AU.'
#        .format(planet_idx, system.types[planet_idx], system.semi_major_axes[planet_idx]))

# times, planet_positions = ... # Your own orbit simulation code
# system.generate_orbit_video(times, planet_positions, filename='orbit_video.xml')


def simulate_engine_performance(
    npb,
):  # npb = number_of_particles_in_box. Code for 1 B and C
    """Function that simulates the performance of one engine in the rocket.
    Takes variablses N(amount of particles in a box)"""
    a = []  # Skal telle hvilke partikler som slipper ut
    nr = []  # Bare til plotting underveis
    rows = 3  # For vectors
    cols = npb

    pos = L * np.random.rand(rows, cols)  # Particle positions
    loc = 0
    scale = np.sqrt(
        const.k_B * T / const.m_H2
    )  # Må bruke for vektorer. Stden stod i boka.
    vel = np.random.normal(
        loc=loc, scale=scale, size=(rows, cols)
    )  # loc = mean, scale = standard deviation(std)
    for m in range(len(t)):  # tidssteg
        pos += dt * vel  # Euler cromer

        x1 = np.where(pos[0] >= L)[0]  # Ser etter kollisjoner for x
        x2 = np.where(pos[0] <= 0)[0]
        y1 = np.where(pos[1] >= L)[0]  # Ser etter kollisjoner for y
        y2 = np.where(pos[1] <= 0)[0]
        z1 = np.where(pos[2] >= L)[0]  # Ser etter kollisjoner for z
        z2 = np.where(pos[2] <= 0)[0]

        for m in range(len(z2)):
            if (
                L / 4 < pos[0][z2[m]] < (3 / 4) * L
            ):  # Sjekker om kollisjonene for z2(xy-planet) egentlig er i utgangshullet
                if L / 4 < pos[1][z2[m]] < (3 / 4) * L:
                    a.append(
                        [pos[0][z2[m]], pos[1][z2[m]], pos[2][z2[m]]]
                    )  # Lagrer partiklene som forsvinner ut. Kan brukes til beregninger
                    # Fjerner partikkelen fra pos, og legger til en ny, uniformt fordelt partikkel, med en vel.
                    if z2[m] not in nr:  # Brukes til plotting
                        nr.append(z2[m])
        z2 = list(z2)
        x1 = list(x1)
        x2 = list(x2)
        y1 = list(y1)
        y2 = list(y2)
        for i in range(len(nr)):  # For plotting
            if nr[i] in z2:
                z2.remove(nr[i])
            if nr[i] in x1:
                x1.remove(nr[i])
            if nr[i] in x2:
                x2.remove(nr[i])
            if nr[i] in y1:
                y1.remove(nr[i])
            if nr[i] in y2:
                y2.remove(nr[i])

        vel[0][x1] = -vel[0][x1]
        vel[0][x2] = -vel[0][
            x2
        ]  # Elastisk støt ved å snu farten til det motsatte i en gitt retning
        vel[1][y1] = -vel[1][y1]
        vel[1][y2] = -vel[1][y2]
        vel[2][z1] = -vel[2][z1]
        vel[2][z2] = -vel[2][z2]

        ax.scatter(pos[0], pos[1], pos[2])  # Plotter rakettmotoren

        # Gjennomsnittlig energi per molekyl
        # Trykk
    # return tbp #thrust per box


x = simulate_engine_performance(200)
plt.show()
