########## Ikke kodemal #############################
import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem
from scipy.stats import norm
import numba as nb
from numba import njit
import math
from P1B import Engine
from P2 import simulate_orbits
import h5py
from part3 import generalized_launch_rocket

utils.check_for_newer_version()

np.random.seed(10)
seed = 57063
system = SolarSystem(seed)
mission = SpaceMission(seed)
Au = 149597870700  # Meters
SM = 1.9891 * 10 ** (30)  # Solar masses in kg
star_mass = system.star_mass  # 0.25361200295275615
star_radius = system.star_radius  # 239265.2554658649
number_of_planets = system.number_of_planets  # 7
semi_major_axes = (
    system.semi_major_axes
)  # [0.06482565 0.0829133  0.36976519 0.22599283 0.16581062 0.58942411 0.04853556]
eccentricities = (
    system.eccentricities
)  # [0.01586665 0.01144546 0.04022341 0.01260794 0.01446333 0.01517603 0.10409536]
masses = (
    system.masses
)  # [3.58554352e-06 1.96851038e-06 2.42297802e-04 4.39924778e-08 3.83112465e-07 1.21242442e-05 3.56579284e-08]
radii = (
    system.radii
)  # [ 6738.5134828   5687.03659344 55556.7550665   1586.28819115 3493.52952637 20342.03369792  1437.99578132]
initial_orbital_angles = (
    system.initial_orbital_angles
)  # [0.         0.74967519 5.85116659 3.04796368 5.27317221 0.10611039 2.36228193] Radians between x-axis and the initial position
aphelion_angles = (
    system.aphelion_angles
)  # [0.         2.07330476 2.44757671 4.07632327 3.93786695 4.11204078 4.92186763] angle between the x-axis and the aphelion direction of each planet, in radians.
rotational_periods = (
    system.rotational_periods
)  # [ 1.12150523  0.77722041  0.59826149  3.07924817 33.21953288  0.61835146 10.17591369] Days
initial_positions = (
    system.initial_positions
)  # [[ 0.06585422  0.06084753  0.32271117 -0.22644101  0.08846641  0.58025758-0.03142163]
# [ 0.          0.05664844 -0.14879135  0.02126361 -0.14086084  0.061803490.03104138]]
initial_velocities = (
    system.initial_velocities
)  # [[  0.          -7.37808042   2.31451309  -0.68985302   6.50085578 -0.48817572 -11.61944718]
# [ 12.23206968   8.10396565   4.89032951  -6.57758159   4.21187235    4.13408761 -10.58597977]]
# G = 4 * (np.pi) ** 2
planet_types = system.types  # ('rock', 'rock', 'gas', 'rock', 'rock', 'gas', 'rock')

distances = mission.measure_distances()

# Fetching data from orbit-files:
filenames = [
    "orbit0.h5",
    "orbit1.h5",
    "orbit2.h5",
    "orbit3.h5",
    "orbit4.h5",
    "orbit5.h5",
    "orbit6.h5",
]
for i, filename in enumerate(filenames):
    h5f = h5py.File(filename, "r")
    globals()[f"orbit_{i}"] = h5f["dataset_1"][:]
    h5f.close()


def spacecraft_triliteration(T, measured_distances):
    """Function to locate position of spacecraft using mesurements to other planets and sun."""
    # finding idx of mesurement time T
    for i, t in enumerate(orbit_0[4]):
        if math.isclose(t, T):
            idx = i
            break
        else:
            continue

    star_pos = np.asarray((0, 0))
    star_distance = measured_distances[-1]
    planet_0_pos = np.asarray((orbit_0[0][idx], orbit_0[1][idx]))
    planet_0_distance = measured_distances[0]
    planet_6_pos = np.asarray((orbit_6[0][idx], orbit_6[1][idx]))
    planet_6_distance = measured_distances[6]
    theta_array = np.arange(0, 2 * np.pi, 10e-6)
    circle_1 = (
        (np.cos(theta_array) * star_distance) + star_pos[0],
        (np.sin(theta_array) * star_distance) + star_pos[1],
    )
    circle_2 = (
        np.cos(theta_array) * planet_0_distance + planet_0_pos[0],
        np.sin(theta_array) * planet_0_distance + planet_0_pos[1],
    )
    circle_3 = (
        np.cos(theta_array) * planet_6_distance + planet_6_pos[0],
        np.sin(theta_array) * planet_6_distance + planet_6_pos[1],
    )
    plt.plot(circle_1[0], circle_1[1], label="circle star")
    plt.plot(circle_2[0], circle_2[1], label="circle homeplanet")
    plt.plot(circle_3[0], circle_3[1], label="circle 3")
    plt.legend()
    plt.show()


distance_list = [0.2, 0.1, 0.4, 0.2, 0.1, 0.7, 0.9, 0.8]
spacecraft_triliteration(0, distance_list)

falcon_engine = Engine(
    N=2 * 10**4, L=3.775 * 10e-8, n_A=1, T=3300, t_c=10e-11, dt=10e-14
)
