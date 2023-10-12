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
from PIL import Image

utils.check_for_newer_version()

np.random.seed(10)
seed = 57063
system = SolarSystem(seed)
mission = SpaceMission(seed)
Au = 149597870700  # Meters
SM = 1.9891 * 10 ** (30)  # Solar masses in kg
sec_per_year = 60 * 60 * 24 * 365
# c = 63239.7263  # Speed of light in Au/yr
c = const.c * (sec_per_year / Au)  # Speed of light in Au/yr
lambda_0 = 656.3  # wavelength of the Hα spectral line from restframe in nanometers
delta_lambda1_sun = mission.star_doppler_shifts_at_sun[0]
delta_lambda2_sun = mission.star_doppler_shifts_at_sun[1]
star_mass = system.star_mass  # 0.25361200295275615
star_radius = system.star_radius  # 239265.2554658649
number_of_planets = system.number_of_planets  # 7

sun_doppler_shift = (
    mission.star_doppler_shifts_at_sun
)  # (-0.020473606152657177, 0.01606904976188539)
star_direction_angles = (
    mission.star_direction_angles
)  # (213.2764103110655, 149.62013634196333)

v_r_sol = c * (
    np.array(sun_doppler_shift) / lambda_0
)  # nanometers / nanometers [-9352.17539636  7340.2101567 ]

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
home_planet_initial_pos = (
    system._initial_positions[0][0],
    system._initial_positions[1][0],
)  # homeplanet initial pos in Au
homeplanet_radius = system._radii[0] * 1000  # homeplanet radius in m


