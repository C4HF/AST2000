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
G = 4 * (np.pi) ** 2  # Gravitational constant using AU
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
planet_types = system.types  # ('rock', 'rock', 'gas', 'rock', 'rock', 'gas', 'rock')
home_planet_initial_pos = (
    system._initial_positions[0][0],
    system._initial_positions[1][0],
)  # homeplanet initial pos in Au
homeplanet_radius = system._radii[0] * 1000  # homeplanet radius in m
"""Parametre"""
m_H2 = const.m_H2
k_B = const.k_B
Au = 149597870700  # Meters
SM = 1.9891 * 10 ** (30)  # Solar masses in kg
G = 6.6743 * (10 ** (-11))  # Gravitational constant
# G = 4 * (np.pi) ** 2  # Gravitational constant for Au
dry_rocket_mass = mission.spacecraft_mass  # kg
crosssection_rocket = mission.spacecraft_area  # m**2
homeplanet_radius = system._radii[0] * 1000  # homeplanet radius in m
homeplanet_mass = system._masses[0] * SM  # homeplanet mass in kg
home_planet_initial_vel = (
    system._initial_velocities[0][0],
    system._initial_velocities[1][0],
)  # homeplanet initital velocity (Au/yr)
home_planet_initial_pos = (
    system._initial_positions[0][0],
    system._initial_positions[1][0],
)  # homeplanet initial pos (Au)
home_planet_rotational_period = system._rotational_periods[
    0
]  # homeplanet rotational period (earth days)

# escape_velocity = np.sqrt((2 * G * homeplanet_mass) / homeplanet_radius)  # m/s
falcon_engine = Engine(
    N=2 * 10**4, L=3.775 * 10e-8, n_A=1, T=3300, t_c=10e-11, dt=10e-14
)
# Fetching data from orbit-files and stores in variables in this script
with np.load("planet_trajectories.npz") as f:
    times = f["times"]
    exact_planet_positions = f["planet_positions"]

for i, planet in enumerate(exact_planet_positions[0]):
    globals()[f"orbit_{i}"] = np.array(
        (
            times,
            exact_planet_positions[0][i],
            exact_planet_positions[1][i],
        )
    )


# Initialize launch #
def rocket_trajectory(total_flight_time, time_of_launch, phi, theta=np.pi / 2):
    (
        altitude,
        vertical_velocity,
        total_time,
        fuel_weight,
        solar_x_pos,
        solar_y_pos,
        solar_x_vel,
        solar_y_vel,
    ) = generalized_launch_rocket(
        falcon_engine,
        fuel_weight=165000,
        launch_theta=np.pi / 2,
        launch_phi=phi,
        launch_time=time_of_launch,
        dt=0.001,
    )

    # finding idx of launch time
    # for i, t in enumerate(orbit_0[0]):
    #     if math.isclose(t, time_of_launch, rel_tol=10e-7):
    #         idx = i
    #         break
    #     else:
    #         continue
    delta_time = orbit_0[0][1] - orbit_0[0][0]
    print(delta_time)
    time_diff = np.abs(orbit_0[0] - time_of_launch)
    least_time_diff = np.min(time_diff)
    idx = np.where(time_diff == least_time_diff)[0]
    start_x_pos = orbit_0[1][idx] + (
        ((homeplanet_radius / Au) * np.cos((np.pi / 2) - theta) * np.cos(phi))
    )  # starting x-postion in Au
    start_y_pos = orbit_0[2][idx] + (
        ((homeplanet_radius / Au) * np.cos((np.pi / 2) - theta) * np.sin(phi))
    )  # starting y_pos in Au
    mission.set_launch_parameters(
        thrust=falcon_engine.thrust,
        mass_loss_rate=falcon_engine.total_fuel_constant,
        initial_fuel_mass=165000,
        estimated_launch_duration=total_time,
        launch_position=[start_x_pos[0], start_y_pos[0]],
        time_of_launch=time_of_launch,
    )
    mission.launch_rocket()
    mission.verify_launch_result([solar_x_pos[0], solar_y_pos[0]])
    time_array = np.arange(0, total_flight_time, delta_time)
    x_pos_arr = np.zeros(len(time_array))
    y_pos_arr = np.zeros(len(time_array))
    x_vel_arr = np.zeros(len(time_array))
    y_vel_arr = np.zeros(len(time_array))

    # Setter initial-betingelser etter launch:
    x_pos_arr[0] = solar_x_pos
    y_pos_arr[0] = solar_y_pos
    x_vel_arr[0] = solar_x_vel
    y_vel_arr[0] = solar_y_vel
    r_planets = np.array(
        [
            [
                orbit_0[1][idx],
                orbit_1[1][idx],
                orbit_2[1][idx],
                orbit_3[1][idx],
                orbit_4[1][idx],
                orbit_5[1][idx],
                orbit_6[1][idx],
            ],
            [
                orbit_0[2][idx],
                orbit_1[2][idx],
                orbit_2[2][idx],
                orbit_3[2][idx],
                orbit_4[2][idx],
                orbit_5[2][idx],
                orbit_6[2][idx],
            ],
        ]
    )
    r_rocket = np.array([x_pos_arr, y_pos_arr])
    v_rocket = np.array([x_vel_arr, y_vel_arr])
    G = 4 * (np.pi) ** 2  # Gravitational constant for Au
    a_sun = (
        (-G * star_mass) * r_rocket[:, 0] / (np.sqrt(np.sum(r_rocket[:, 0] ** 2))) ** 3
    )  # Sets initial acceleration from sun according to N.2 law
    # a_planets = np.sum((G * masses * (r_rocket[:, 0] - r_planets[:, idx]))) / (
    #     np.abs(r_rocket[:, 0] - r_planets[:, idx]) ** 3
    # )  # Sets initial acceleration from planets according to N.2 law
    # acc_old = a_sun + a_planets
    acc_old = a_sun
    for i in range(0, len(time_array) - 1):
        r_rocket[:, i + 1] = (
            r_rocket[:, i]
            + v_rocket[:, i] * delta_time
            + (acc_old * delta_time**2) / 2
        )
        a_sun = (
            (-G * star_mass)
            * r_rocket[:, i + 1]
            / (np.sqrt(np.sum(r_rocket[:, i + 1] ** 2)) ** 3)
        )  # Sets initial acceleration from sun according to N.2 law
        # a_planets = np.sum(
        #     (G * masses * (r_rocket[:, i + 1] - r_planets[:, i + 1]))
        # ) / (
        #     np.abs(r_rocket[:, i + 1] - r_planets[:, i + 1]) ** 3
        # )  # Sets initial acceleration from planets according to N.2 law
        # acc_new = a_sun + a_planets
        acc_new = a_sun
        v_rocket[:, i + 1] = v_rocket[:, i] + (1 / 2) * (acc_old + acc_new) * delta_time
        acc_old = acc_new

    return time_array, r_rocket, v_rocket


(time_array, r_rocket, v_rocket) = rocket_trajectory(0.03, 0, np.pi)
plt.plot(r_rocket[0, :], r_rocket[1, :])
plt.plot(orbit_0[1], orbit_0[2])
plt.show()
"""
Leapfrog:
        x_pos[i] = (
            x_pos[i - 1] + (x_vel[i - 1] * dt) + ((x_acc_old * dt**2) / 2)
        )  # updating x-pos
        y_pos[i] = (
            y_pos[i - 1] + (y_vel[i - 1] * dt) + ((y_acc_old * dt**2) / 2)
        )  # updating y-pos

        x_acc_new = (gamma * x_pos[i]) / (
            np.sqrt((x_pos[i] ** 2) + (y_pos[i] ** 2))
        ) ** 3  # setting new x-acceleration using the position of in the last iteration
        y_acc_new = (gamma * y_pos[i]) / (
            np.sqrt(x_pos[i] ** 2 + y_pos[i] ** 2)
        ) ** 3  # setting new y-acceleration using the position of in the last iteration

         x_vel[i] = (
            x_vel[i - 1] + (1 / 2) * (x_acc_old + x_acc_new) * dt
        )  # updating x-velocity
        y_vel[i] = (
            y_vel[i - 1] + (1 / 2) * (y_acc_old + y_acc_new) * dt
        )  # updating y-velocity
        x_acc_old = x_acc_new  # setting old x-aceleration to new x-acceleration to prepare for next iteration
        y_acc_old = y_acc_new  # setting old y-aceleration to new y-acceleration to prepare for next iteration
"""

# rocket_trajectory(1, 0)

# def rocket_path(t0, r0, v0, T, dt):
#     "Utilises leapfrog and newtons 2.law to calculate the rockets path given initial values of the rocket"
#     "t0 in years"
#     "r0 in AU"
#     "v0 in AU / yr"
#     "T in years"
#     "dt in years"
#     N = int(T / dt)  # Defines length of arrays
#     t = np.zeros(N)  # Time array
#     t[0] = t0  # Sets first values of arrays
#     r = np.zeros((2, N))  # Position array
#     r[0] = r0
#     v = np.zeros((2, N))  # Velocity array
#     v[0] = v0
#     a = np.zeros((2, N))  # Acceleration array
#     a_0_sun = (
#         -G * (masses[0] * star_mass) * r0 / (np.abs(r0) ** 3)
#     )  # Sets initial acceleration from sun according to N.2 law
#     a_0_planets = -np.sum(
#         (
#             G
#             * masses[0]
#             * masses[1:]
#             * (r0 - np.array([initial_positions[0, 1:], initial_positions[1, 1:]]))
#         )
#         / (
#             np.abs(
#                 r0 - np.array([initial_positions[0, 1:], initial_positions[1, 1:]]) ** 3
#             )
#         )
#     )  # Sets initial acceleration from planets according to N.2 law
#     a[0] = a_0_sun + a_0_planets  # Sets first value of acceleration array
#     return print(r, v, a)
#     # return t_final, r_final, v_final
