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
from P1B import calculate_needed_fuel
from PIL import Image
from scipy import interpolate

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
# G = 6.6743 * (10 ** (-11))  # Gravitational constant
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
orbits = np.array([orbit_0, orbit_1, orbit_2, orbit_3, orbit_4, orbit_5, orbit_6])


def rocket_trajectory(
    initial_time,
    initial_x_pos,
    initial_y_pos,
    initial_x_vel,
    initial_y_vel,
    total_flight_time,
    time_step,
):
    time_array = np.arange(initial_time, initial_time + total_flight_time, time_step)
    x_pos_arr = np.zeros(len(time_array))
    y_pos_arr = np.zeros(len(time_array))
    x_vel_arr = np.zeros(len(time_array))
    y_vel_arr = np.zeros(len(time_array))

    x_pos_arr[0] = initial_x_pos
    y_pos_arr[0] = initial_y_pos
    x_vel_arr[0] = initial_x_vel
    y_vel_arr[0] = initial_y_vel

    interpolated_orbits = [[], []]

    for orbit in orbits:
        f = interpolate.interp1d(orbit[0], orbit[1:3], axis=-1)
        t = time_array
        x, y = f(t)
        interpolated_orbits[0].append(x)
        interpolated_orbits[1].append(y)

    r_planets = np.array(interpolated_orbits)

    r_rocket = np.array([[x_pos_arr], [y_pos_arr]])
    v_rocket = np.array([[x_vel_arr], [y_vel_arr]])

    G = 4 * (np.pi) ** 2  # Gravitational constant for Au
    a_sun = ((-G * star_mass) * r_rocket[:, :, 0]) / (
        np.sqrt(np.sum(r_rocket[:, :, 0] ** 2))
    ) ** 3  # Sets initial acceleration from sun according to N.2 law
    a_planets = -np.sum(
        ((G * masses * np.sqrt(np.sum((r_rocket[:, :, 0] - r_planets[:, :, 0]) ** 2))))
        / (np.sqrt(np.sum((r_rocket[:, :, 0] - r_planets[:, :, 0]) ** 2)) ** 3)
    )  # Sets initial acceleration from planets according to N.2 law
    acc_old = a_sun + a_planets
    # acc_old = a_planets
    for i in range(0, len(time_array) - 1):
        r_rocket[:, :, i + 1] = (
            r_rocket[:, :, i]
            + v_rocket[:, :, i] * time_step
            + (acc_old * time_step**2) / 2
        )

        a_sun = (
            (-G * star_mass)
            * r_rocket[:, :, i + 1]
            / (np.sqrt(np.sum(r_rocket[:, :, i + 1] ** 2)) ** 3)
        )  # Sets initial acceleration from sun according to N.2 law
        a_planets = -np.sum(
            (
                (
                    G
                    * masses
                    * np.sqrt(np.sum((r_rocket[:, :, i] - r_planets[:, :, i]) ** 2))
                )
            )
            / (np.sqrt(np.sum((r_rocket[:, :, i] - r_planets[:, :, i]) ** 2)) ** 3)
        )  # Sets initial acceleration from planets according to N.2 law
        acc_new = a_sun + a_planets
        # acc_new = a_planets
        v_rocket[:, :, i + 1] = (
            v_rocket[:, :, i] + (1 / 2) * (acc_old + acc_new) * time_step
        )
        acc_old = acc_new

    return time_array, r_rocket, v_rocket, r_planets


"""Code to find best launch time and launch phi."""
"""
launch_times = np.linspace(0, 2, 1000)
launch_phis = np.linspace(0, 2 * np.pi, 4)

shortest_dist = 100000
best_launch_time = 0
best_launch_phi = 0
success = []

for launch_time in launch_times:
    for launch_phi in launch_phis:
        print(f"Checking launch time: {launch_time} and launch phi: {launch_phi}...")
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
            launch_phi=launch_phi,
            launch_time=launch_time,
            dt=0.01,
        )

        (time_array, r_rocket, v_rocket, r_planets) = rocket_trajectory(
            launch_time + (total_time / sec_per_year),
            solar_x_pos,
            solar_y_pos,
            solar_x_vel,
            solar_y_vel,
            total_flight_time=2.9 - launch_time,
            time_step=10e-6,
        )

    dist_array = np.sqrt(
        (r_rocket[0, 0, :] - r_planets[0, 1, :]) ** 2
        + (r_rocket[1, 0, :] - r_planets[1, 1, :]) ** 2
    )
    dist = np.min(dist_array)
    idx = np.where(dist_array == dist)[0]
    dist_to_star = np.sqrt(r_rocket[0, 0, idx] ** 2 + r_rocket[1, 0, idx] ** 2)
    gravitational_capture_dist = dist_to_star * np.sqrt(masses[1] / (10 * star_mass))
    if dist < gravitational_capture_dist:
        print("******************")
        print("Success!")
        print(f"Launchtime: {launch_time}")
        print(f"Launch_phi: {launch_phi}")
        print("******************")
        success.append((launch_time, launch_phi, dist))
        # break
    if dist < shortest_dist:
        shortest_dist = dist
        best_launch_time = launch_time
        best_launch_phi = launch_phi

    print("---------")
    print(f"Current shortest dist: {shortest_dist}")
    print("---------")
for i in range(0, len(r_rocket[0, 0]), 10000):
    dist = np.sqrt(
        (r_rocket[0, 0, i] - r_planets[0, 1, i]) ** 2
        + (r_rocket[1, 0, i] - r_planets[1, 1, i]) ** 2
    )
    dist_to_star = np.sqrt(r_rocket[0, 0, i] ** 2 + r_rocket[1, 0, i] ** 2)
    gravitational_capture_dist = dist_to_star * np.sqrt(masses[1] / (10 * star_mass))
    if dist < gravitational_capture_dist:
        print("Success!")
        print(f"Launchtime: {launch_time}")
        print(f"Launch_phi: {launch_phi}")
        break
    if dist < shortest_dist:
        shortest_dist = dist
        best_launch_time = launch_time
        best_launch_phi = launch_phi

print("---------")
print("Finished. Found:")
print(f"Current shortest dist {shortest_dist}")
print(f"Launchtime: {launch_time}")
print(f"Launch_phi: {launch_phi}")
print(f"Shortest_dist (Au): {shortest_dist}")
print(f"Success: {success})")
print("-----------")
"""
""" 
Results:
Searching parameters: # launch_times = np.linspace(0, 2, 1000), launch_phis = np.linspace(0, 2 * np.pi, 4), 
with dt=0.01 and timestep=10e-5 found best: (0.8368368368368369, 6.283185307179586, 8.240883598534517e-06)
"""
best_launch_time_dt05 = 0.8368368368368369
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
    launch_phi=6.283185307179586,
    launch_time=best_launch_time_dt05,
    dt=0.01,
)
print(f"After launch v: {solar_x_vel}, {solar_y_vel}")

(time_array, r_rocket1, v_rocket1, r_planets) = rocket_trajectory(
    best_launch_time_dt05 + (total_time / sec_per_year),
    solar_x_pos,
    solar_y_pos,
    solar_x_vel,
    solar_y_vel,
    total_flight_time=3.0 - (best_launch_time_dt05 + (total_time / sec_per_year)),
    time_step=10e-6,
)
dist_array = np.sqrt(
    (r_rocket1[0, 0, :] - r_planets[0, 1, :]) ** 2
    + (r_rocket1[1, 0, :] - r_planets[1, 1, :]) ** 2
)
dist = np.min(dist_array)
idx1 = np.where(dist_array == dist)[0]
time_of_least_distance = time_array[idx1]

"Code to calculate velocity vectors, used to calculate boost vector:"
v_planet1_shortest_dist = np.array(
    (r_planets[:, 1, idx1] - r_planets[:, 1, idx1 - 1])
    / (time_array[idx1] - time_array[idx1 - 1])
)
dist_to_star = np.sqrt(r_rocket1[0, 0, idx1] ** 2 + r_rocket1[1, 0, idx1] ** 2)
gravitational_capture_dist = dist_to_star * np.sqrt(masses[1] / (10 * star_mass))
v_rocket1_shortest_dist = np.array((r_rocket1[0, 0, idx1][0], r_rocket1[1, 0, idx1][0]))
r_rocket1_shortest_dist = (r_rocket1[0, 0, idx1][0], r_rocket1[1, 0, idx1][0])
r_planet1_shortest_dist = np.array((r_planets[0, 0, idx1], r_planets[1, 1, idx1]))
r_radial = r_rocket1_shortest_dist - r_planet1_shortest_dist
unit_vector_1 = v_rocket1_shortest_dist / np.linalg.norm(v_rocket1_shortest_dist)
unit_vector_2 = r_radial / np.linalg.norm(r_radial)
dot_product = np.dot(unit_vector_1, unit_vector_2)
angle = np.arccos(dot_product)
v_rocket1_tangential = v_rocket1_shortest_dist * np.sin(angle)
v_rocket1_radial = -v_rocket1_shortest_dist * np.cos(angle)

# Finding angle of first boost
"""Code to find boost-angle"""
"""
boost_angles = np.linspace(-np.pi / 2, np.pi / 2, 10)
best_angle = 0
least_distance = 10000

for i, boost_angle in enumerate(boost_angles):
    boost_rocket_x = v_rocket1_radial[0] * np.cos(boost_angle) - v_rocket1_radial[
        1
    ] * np.cos(boost_angle)
    boost_rocket_y = v_rocket1_radial[0] * np.sin(boost_angle) + v_rocket1_radial[
        1
    ] * np.cos(boost_angle)
    (time_array, r_rocket, v_rocket, r_planets) = rocket_trajectory(
        time_of_least_distance,
        r_rocket1[0, 0, idx1],
        r_rocket1[1, 0, idx1],
        v_rocket1[0, 0, idx1] + boost_rocket_x * 100,
        v_rocket1[1, 0, idx1] + boost_rocket_y * 100,
        total_flight_time=3 - time_of_least_distance,
        time_step=10e-6,
    )

    dist_array = np.sqrt(
        (r_rocket[0, 0, :] - r_planets[0, 1, :]) ** 2
        + (r_rocket[1, 0, :] - r_planets[1, 1, :]) ** 2
    )

    dist = np.min(dist_array)
    idx2 = np.where(dist_array == dist)[0]
    time_of_least_distance2 = time_array[idx2]
    dist_to_star = np.sqrt(r_rocket[0, 0, idx2] ** 2 + r_rocket[1, 0, idx2] ** 2)
    gravitational_capture_dist = dist_to_star * np.sqrt(masses[1] / (10 * star_mass))
    print(f"Checking {i}/{len(boost_angles)}...")
    # plt.plot(r_rocket[0, 0, :], r_rocket[1, 0, :], label=f"boost {boost_angle}")
    if dist <= gravitational_capture_dist:
        least_distance = dist
        best_angle = boost_angle
        print("*************")
        print("Success!")
        print(f"Current best angle: {best_angle} ")
        # print(f"Current shortest dist: {least_distance}")
        # print(f"Time of least distance: {time_of_least_distance2}")
        print(f"*************")
    if dist < least_distance:
        least_distance = dist
        best_angle = boost_angle
        print("*************")
        print(f"Current best angle: {best_angle} ")
        print(f"Current shortest dist: {least_distance}")
        print(f"Time of least distance: {time_of_least_distance2}")
        print(f"*************")
# for i in range(7):
#     plt.plot(r_planets[0, i, :], r_planets[1, i, :], label=f"orbit{i}")
# plt.plot(r_rocket1[0, 0, :], r_rocket1[1, 0, :], label="Rocket trajectory before boost")
# plt.scatter(r_rocket1[0, 0, -1], r_rocket1[1, 0, -1], label="Rocket before boost")
# plt.legend()
# plt.show()
"""
""""
*************
Results
Current best angle: -0.8726646259971648
Current shortest dist: 3.0515403005546108e-05
Time of least distance: [2.652451]
*************
"""
best_angle = -0.8726646259971648
boost_rocket_x = v_rocket1_radial[0] * np.cos(best_angle) - v_rocket1_radial[
    1
] * np.cos(best_angle)
boost_rocket_y = v_rocket1_radial[0] * np.sin(best_angle) + v_rocket1_radial[
    1
] * np.cos(best_angle)

fuel_boost1 = calculate_needed_fuel(
    falcon_engine,
    dry_rocket_mass + fuel_weight,
    np.linalg.norm(boost_rocket_x * 100 + boost_rocket_y**100),
    dt=0.00001,
)
print(
    f"T: {time_array[idx1]} yr. Boost 1: {np.linalg.norm(boost_rocket_x * 100 + boost_rocket_y**100)}, ({boost_rocket_x * 100},{boost_rocket_y**100}) Au/yr, v: ({v_rocket1[0, 0, idx1] + boost_rocket_x * 100}, {v_rocket1[1, 0, idx1] + boost_rocket_y * 100})"
)
(time_array, r_rocket, v_rocket, r_planets) = rocket_trajectory(
    time_of_least_distance,
    r_rocket1[0, 0, idx1],
    r_rocket1[1, 0, idx1],
    v_rocket1[0, 0, idx1] + boost_rocket_x * 100,
    v_rocket1[1, 0, idx1] + boost_rocket_y * 100,
    total_flight_time=3 - time_of_least_distance,
    time_step=10e-6,
)

dist_array = np.sqrt(
    (r_rocket[0, 0, :] - r_planets[0, 1, :]) ** 2
    + (r_rocket[1, 0, :] - r_planets[1, 1, :]) ** 2
)

dist = np.min(dist_array)
idx2 = np.where(dist_array == dist)[0]
time_of_least_distance2 = time_array[idx2]
dist_to_star = np.sqrt(r_rocket[0, 0, idx2] ** 2 + r_rocket[1, 0, idx2] ** 2)
# gravitational_capture_dist = dist_to_star * np.sqrt(masses[1] / (10 * star_mass))

dist_to_star_array = np.sqrt(r_rocket[0, 0, :] ** 2 + r_rocket[1, 0, :] ** 2)
plt.plot(time_array, dist_array, label="Dist to planet")
plt.plot(
    time_array,
    dist_to_star_array * np.sqrt(masses[1] / (10 * star_mass)),
    label="Gravitational capture",
)
plt.scatter(time_array[idx2], dist_array[idx2], label="Shortest distance")
plt.xlabel("T", fontsize=20)
plt.ylabel("Au", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()
plt.title("Distance to target planet vs. gravitational capture distance")
plt.show()

plt.plot(
    r_rocket[0, 0, 0 : int(idx2 + 1)],
    r_rocket[1, 0, 0 : int(idx2 + 1)],
    label="Rocket trajectory after correctional boost",
)
plt.plot(
    r_rocket1[0, 0, 0 : int(idx1 + 1)],
    r_rocket1[1, 0, 0 : int(idx1 + 1)],
    label="Rocket trajectory after launch",
)
for i in range(7):
    plt.plot(r_planets[0, i, :], r_planets[1, i, :], label=f"orbit{i}")

plt.scatter(
    r_rocket[0, 0, idx2],
    r_rocket[1, 0, idx2],
    label="Shortest dist rocket",
)
plt.scatter(
    r_planets[0, 1, idx2],
    r_planets[1, 1, idx2],
    label="Shortest dist planet",
)
plt.scatter(
    r_rocket1[0, 0, idx1],
    r_rocket1[1, 0, idx1],
    label="Rocket before boost",
)
plt.scatter(0, 0, label="Sun")
plt.scatter(r_rocket1[0, 0, 0], r_rocket1[1, 0, 0], label="Rocket after launch")


gravitational_capture_dist_array = np.sqrt(
    r_rocket[0, 0, :] ** 2 + r_rocket[1, 0, :] ** 2
) * np.sqrt(masses[1] / (10 * star_mass))

v2_planet1_shortest_dist = np.array(
    (r_planets[:, 1, idx2] - r_planets[:, 1, idx2 - 1])
    / (time_array[idx2] - time_array[idx2 - 1])
)

"Code to calculate velocity vectors, used to calculate boost vector"
v_rocket2_shortest_dist = np.array((v_rocket[0, 0, idx2][0], v_rocket[1, 0, idx2][0]))
r_rocket2_shortest_dist = (r_rocket[0, 0, idx2][0], r_rocket[1, 0, idx2][0])
r_planet2_shortest_dist = np.array((r_planets[0, 1, idx2], r_planets[1, 1, idx2]))
r_radial = r_rocket2_shortest_dist - r_planet2_shortest_dist
unit_vector_1 = v_rocket2_shortest_dist / np.linalg.norm(v_rocket2_shortest_dist)
unit_vector_2 = r_radial / np.linalg.norm(r_radial)
dot_product = np.dot(unit_vector_1, unit_vector_2)
angle = np.arccos(dot_product)
v_rocket2_tangential = v_rocket2_shortest_dist * np.sin(angle)
v_rocket2_radial = -v_rocket2_shortest_dist * np.cos(angle)
abs_v_stable = np.sqrt((G * masses[1]) / dist)
abs_v_tangential = np.linalg.norm(v_rocket2_tangential)
r = abs_v_stable / abs_v_tangential
# v_stable = v2_planet1_shortest_dist[:, 0] + v_rocket2_tangential * r
v_stable = v2_planet1_shortest_dist + v_rocket2_tangential * r
v_injection = v_stable - v_rocket1_shortest_dist

(time_array, r_rocket3, v_rocket3, r_planets) = rocket_trajectory(
    time_of_least_distance,
    r_rocket[0, 0, idx2],
    r_rocket[1, 0, idx2],
    v_stable[0],
    v_stable[1],
    total_flight_time=3 - time_of_least_distance2,
    time_step=10e-6,
)
v_boost = np.linalg.norm((v_rocket[:, 0, idx2] - v_stable))
fuel_injection = calculate_needed_fuel(
    falcon_engine, dry_rocket_mass + (fuel_weight - fuel_boost1), v_boost, dt=0.00001
)

print(
    f"T: {time_of_least_distance2} yr. Boost: ({v_rocket2_shortest_dist[0] - v_stable[0]}, {v_rocket2_shortest_dist[1]-v_stable[1]}) Au/yr, V: ({v_stable[0]}, {v_stable[1]})"
)
plt.plot(
    r_rocket3[0, 0, :],
    r_rocket3[1, 0, :],
    label="Rocket trajectory after injection",
)
plt.xlabel("Au", fontsize=20)
plt.ylabel("Au", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=10)
plt.title("Rocket trajectory towards target planet", fontsize=20)
plt.show()

print("-------------")
print(f"Fuel before launch:{165000} kg")
print(f"Fuel after launch:{fuel_weight} kg")
print(f"Fuel after first boost: {fuel_weight - fuel_boost1} kg")
print(f"Fuel after injection: {fuel_weight - fuel_boost1-fuel_injection} kg")
print("------------")
