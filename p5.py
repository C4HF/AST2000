########## Ikke kodemal #############################
import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem
from ast2000tools.shortcuts import SpaceMissionShortcuts
from scipy.stats import norm
from P1B import Engine
from part3 import generalized_launch_rocket
from P1B import calculate_needed_fuel
from p4 import spacecraft_triliteration, calculate_velocity_from_doppler, find_phi
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
    """Function to calculate rocket trajectory while coasting in SolarSystem. Takes initial time, position, velocity,
    desired coast-time and the timestep for the simulation (a smaller timestep increases accuracy). The function
    interpolates the planet-orbits and use the posistion of the planets in each timestep to calculate the trajectory of the
    rocket using a leapfrog-loop and Newtons gravitational formula. Returns the trajectory of rocket including time, velocity
    and position, aswell as the interpolated planet orbits.
    .
    """
    time_array = np.arange(
        initial_time, initial_time + total_flight_time, time_step
    )  # Creates the time_array of simulation
    x_pos_arr = np.zeros(
        len(time_array)
    )  # Empty x-position-array to be filled with values in simulation
    y_pos_arr = np.zeros(
        len(time_array)
    )  # Empty y-position-array to be filled with values in simulation
    x_vel_arr = np.zeros(
        len(time_array)
    )  # Empty x-velocity-array to be filled with values in simulation
    y_vel_arr = np.zeros(
        len(time_array)
    )  # Empty y-veloctiy-array to be filled with values in simulation

    x_pos_arr[0] = initial_x_pos  # Setting initial x-position
    y_pos_arr[0] = initial_y_pos  # Setting initial y-position
    x_vel_arr[0] = initial_x_vel  # Setting initial x-velocity
    y_vel_arr[0] = initial_y_vel  # Setting initial y-velocity

    interpolated_orbits = [[], []]  # Empty list to be filled with interpolated orbits

    # Loop to interpolate simulated orbits to fit shape of time_array.
    for orbit in orbits:
        f = interpolate.interp1d(
            orbit[0], orbit[1:3], axis=-1
        )  # Using scipy.interpolate to interpolate orbits
        t = time_array
        x, y = f(
            t
        )  # Using interpolation-function to calculate planet-postions during the timesteps in time_array
        interpolated_orbits[0].append(x)  # adding to list
        interpolated_orbits[1].append(y)  # adding to list

    r_planets = np.array(
        interpolated_orbits
    )  # Turning interpolated_orbits-list to array

    r_rocket = np.array(
        [[x_pos_arr], [y_pos_arr]]
    )  # Empty pos-array for rocket position, to be filled with values from simulation
    v_rocket = np.array(
        [[x_vel_arr], [y_vel_arr]]
    )  # Empty vel-array for rocket position, to be filled with values from simulation

    G = 4 * (np.pi) ** 2  # Gravitational constant for Au
    a_sun = ((-G * star_mass) * r_rocket[:, :, 0]) / (
        np.sqrt(np.sum(r_rocket[:, :, 0] ** 2))
    ) ** 3  # Sets initial acceleration from sun according to N.2 law
    a_planets = -np.sum(
        (
            (
                G
                * system.masses
                * np.sqrt(np.sum((r_rocket[:, :, 0] - r_planets[:, :, 0]) ** 2))
            )
        )
        / (np.sqrt(np.sum((r_rocket[:, :, 0] - r_planets[:, :, 0]) ** 2)) ** 3)
    )  # Sets initial acceleration from planets according to N.2 law
    acc_old = a_sun + a_planets
    # Leapfrog-loop
    for i in range(0, len(time_array) - 1):
        r_rocket[:, :, i + 1] = (
            r_rocket[:, :, i]
            + v_rocket[:, :, i] * time_step
            + (acc_old * time_step**2) / 2
        )  # Rocket pos at time i+1

        a_sun = (
            (-G * star_mass)
            * r_rocket[:, :, i + 1]
            / (np.sqrt(np.sum(r_rocket[:, :, i + 1] ** 2)) ** 3)
        )  # Sets initial acceleration from sun according to N.2 law
        a_planets = -np.sum(
            (
                (
                    G
                    * system.masses
                    * np.sqrt(np.sum((r_rocket[:, :, i] - r_planets[:, :, i]) ** 2))
                )
            )
            / (np.sqrt(np.sum((r_rocket[:, :, i] - r_planets[:, :, i]) ** 2)) ** 3)
        )  # Sets initial acceleration from planets according to N.2 law
        acc_new = (
            a_sun + a_planets
        )  # Setting new acceleration to calculate velocity change
        v_rocket[:, :, i + 1] = (
            v_rocket[:, :, i] + (1 / 2) * (acc_old + acc_new) * time_step
        )  # Calculating velocty of rocket in timestep i+1
        acc_old = acc_new

    return time_array, r_rocket, v_rocket, r_planets


#################################################################
# #   Bruteforcing to find best launchtime/launchangle        # #
#################################################################
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
Searching parameters: # launch_times = np.linspace(0, 2, 1000), launch_phis = np.linspace(0, 2 * np.pi, 4), 
with dt=0.01 and timestep=10e-5 
Best result: (Time (yr): 0.8368368368368369, Angle (radians):6.283185307179586, Min. dist. (Au): 8.240883598534517e-06)
"""
best_launch_time_dt05 = 0.8368368368368369


#################################################################
# #   Launching using best launchtime and timestep 10e-6      # #
#################################################################
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

#################################################################
# #              Updating space_mission-instance              # #
#################################################################
time_diff = np.abs(orbit_0[0] - best_launch_time_dt05)
least_time_diff = np.min(time_diff)
idx_ = np.where(time_diff == least_time_diff)[0]
initial_solar_x_pos = orbit_0[1][idx_] + (
    (
        (homeplanet_radius / Au)
        * np.cos((np.pi / 2) - np.pi / 2)
        * np.cos(6.283185307179586)
    )
)  # starting y-postion in Au
initial_solar_y_pos = orbit_0[2][idx_] + (
    (
        (homeplanet_radius / Au)
        * np.cos((np.pi / 2) - np.pi / 2)
        * np.sin(6.283185307179586)
    )
)  # starting y-postion in Au
mission.set_launch_parameters(
    thrust=falcon_engine.thrust,
    mass_loss_rate=falcon_engine.total_fuel_constant,
    initial_fuel_mass=165000,
    estimated_launch_duration=446.7099999963486,
    # launch_position=[initial_solar_x_pos[0], initial_solar_y_pos[0]],
    launch_position=[initial_solar_x_pos[0], initial_solar_y_pos[0]],
    time_of_launch=orbit_0[0, idx_],
)
mission.launch_rocket()
mission.verify_launch_result(
    (solar_x_pos[0], solar_y_pos[0])
)  # verifies that the calculated launch-results are correct
distances = mission.measure_distances()
takenimage = mission.take_picture()
mesured_dopplershifts = mission.measure_star_doppler_shifts()
pos_after_launch = spacecraft_triliteration(
    best_launch_time_dt05 + total_time / sec_per_year, distances
)
vel_after_launch = calculate_velocity_from_doppler(
    mesured_dopplershifts[0], mesured_dopplershifts[1]
)
angle_after_launch = find_phi("sky_picture.png")
mission.verify_manual_orientation(
    pos_after_launch, vel_after_launch, angle_after_launch
)


#################################################################
# #                 Rocket trajectory 1                       # #
#################################################################
# Simulating rocket trajectory after launch. ##

(time_array1, r_rocket1, v_rocket1, r_planets1) = rocket_trajectory(
    best_launch_time_dt05 + (total_time / sec_per_year),
    solar_x_pos,
    solar_y_pos,
    solar_x_vel,
    solar_y_vel,
    total_flight_time=3.0 - (best_launch_time_dt05 + (total_time / sec_per_year)),
    time_step=10e-5,
)

## Finding minimal distance and time of minimal distance ##
dist_array = np.sqrt(
    (r_rocket1[0, 0, :] - r_planets1[0, 1, :]) ** 2
    + (r_rocket1[1, 0, :] - r_planets1[1, 1, :]) ** 2
)
dist1 = np.min(dist_array)
idx1 = np.where(dist_array == dist1)[0]
time_of_least_distance1 = time_array1[idx1]

## Calculating different position and velocity-vectors ##
v_planet1_shortest_dist = np.array(
    (r_planets1[:, 1, idx1] - r_planets1[:, 1, idx1 - 1])
    / (time_array1[idx1] - time_array1[idx1 - 1])
)
dist_to_star1 = np.sqrt(r_rocket1[0, 0, idx1] ** 2 + r_rocket1[1, 0, idx1] ** 2)
gravitational_capture_dist = dist_to_star1 * np.sqrt(
    system.masses[1] / (10 * star_mass)
)
v_rocket1_shortest_dist = np.array((r_rocket1[0, 0, idx1][0], r_rocket1[1, 0, idx1][0]))
r_rocket1_shortest_dist = (r_rocket1[0, 0, idx1][0], r_rocket1[1, 0, idx1][0])
r_planet1_shortest_dist = np.array((r_planets1[0, 0, idx1], r_planets1[1, 1, idx1]))
r_radial = r_rocket1_shortest_dist - r_planet1_shortest_dist
unit_vector_1 = v_rocket1_shortest_dist / np.linalg.norm(v_rocket1_shortest_dist)
unit_vector_2 = r_radial / np.linalg.norm(
    r_radial
)  # radial position-vector relative to planet
dot_product = np.dot(
    unit_vector_1, unit_vector_2
)  # rocket-velocity-vector in the x/y-plane
angle = np.arccos(
    dot_product
)  # <---- Angle between rocket-velocity-vector and radial position-vector relative to planet. ####
v_rocket1_tangential = v_rocket1_shortest_dist * np.sin(angle)
v_rocket1_radial = -v_rocket1_shortest_dist * np.cos(angle)


#################################################################
# #  Bruteforcing to find best angle for correctional boost   # #
#################################################################
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
Results:
Best angle (radians): -0.8726646259971648
Shortest dist (Au): 3.0515403005546108e-05
*************
"""

best_angle = -0.8726646259971648

boost_rocket_x = (
    v_rocket1_radial[0] * np.cos(best_angle) - v_rocket1_radial[1] * np.cos(best_angle)
) * 100  # Rocket-boost x-component
boost_rocket_y = (
    v_rocket1_radial[0] * np.sin(best_angle) + v_rocket1_radial[1] * np.cos(best_angle)
) * 100  # Rocket-boost y-component

# Calculating needed fuel for first boost
fuel_boost1 = calculate_needed_fuel(
    falcon_engine,
    dry_rocket_mass + fuel_weight,
    np.linalg.norm(boost_rocket_x + boost_rocket_y),
    dt=0.001,
)

#################################################################
# #                 Rocket trajectory 2                       # #
#################################################################
## Simulating rocket trajectory after first correctional boost ##
(time_array2, r_rocket2, v_rocket2, r_planets2) = rocket_trajectory(
    time_of_least_distance1,
    r_rocket1[0, 0, idx1],
    r_rocket1[1, 0, idx1],
    v_rocket1[0, 0, idx1] + boost_rocket_x,
    v_rocket1[1, 0, idx1] + boost_rocket_y,
    total_flight_time=3 - time_of_least_distance1,
    time_step=10e-6,
)

## Finding minimal distance and time of minimal distance ##
dist_array2 = np.sqrt(
    (r_rocket2[0, 0, :] - r_planets2[0, 1, :]) ** 2
    + (r_rocket2[1, 0, :] - r_planets2[1, 1, :]) ** 2
)
dist2 = np.min(dist_array)
idx2 = np.where(dist_array == dist2)[0]
time_of_least_distance2 = time_array2[idx2]
dist_to_star2 = np.sqrt(r_rocket2[0, 0, idx2] ** 2 + r_rocket2[1, 0, idx2] ** 2)
gravitational_capture_dist = dist_to_star2 * np.sqrt(
    system.masses[1] / (10 * star_mass)
)  # Calculating gravitational capture distance of planet (l)
dist_to_star_array2 = np.sqrt(r_rocket2[0, 0, :] ** 2 + r_rocket2[1, 0, :] ** 2)


#################################################################
# # Plotting dist. from rocket traj. 2 to planet 1 with l     # #
#################################################################
plt.plot(time_array2, dist_array2, label="Dist to planet")
plt.plot(
    time_array2,
    dist_to_star_array2 * np.sqrt(system.masses[1] / (10 * star_mass)),
    label="Gravitational capture",
)
plt.scatter(
    time_array2[idx2],
    dist_array[idx2],
    label=f"Shortest distance: {dist2:.2e}",
)
plt.xlabel("Yr", fontsize=20)
plt.ylabel("Au", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.title("Distance to target planet vs. gravitational capture distance", fontsize=20)
plt.show()


#################################################################
# #     Calculating boost vectors for injection maneuvre      # #
#################################################################
v2_planet1_shortest_dist = np.array(
    (r_planets2[:, 1, idx2] - r_planets2[:, 1, idx2 - 1])
    / (time_array2[idx2] - time_array2[idx2 - 1])
)  # planet 1 velcotiry at shortest distance
v_rocket2_shortest_dist = np.array(
    (v_rocket2[0, 0, idx2][0], v_rocket2[1, 0, idx2][0])
)  # rocket velcotiy at shortest distance
r_rocket2_shortest_dist = (r_rocket2[0, 0, idx2][0], r_rocket2[1, 0, idx2][0])
r_planet2_shortest_dist = np.array((r_planets2[0, 1, idx2], r_planets2[1, 1, idx2]))
r_radial = (
    r_rocket2_shortest_dist - r_planet2_shortest_dist
)  # radial position vector at shortest distance
unit_vector_1 = v_rocket2_shortest_dist / np.linalg.norm(
    v_rocket2_shortest_dist
)  # Rocket velocity-unit-vector at shortest ditance
unit_vector_2 = r_radial / np.linalg.norm(
    r_radial
)  # Rocket radial-postion-unit-vector at shortest ditance
dot_product = np.dot(unit_vector_1, unit_vector_2)
angle = np.arccos(
    dot_product
)  # <---- Angle between radial-vector and rocket velocity-vector
v_rocket2_tangential = v_rocket2_shortest_dist * np.sin(
    angle
)  # rocket tangential velcocity at shortest distance
v_rocket2_radial = -v_rocket2_shortest_dist * np.cos(
    angle
)  # rocket radial velcocity at shortest distance
abs_v_stable = np.sqrt((G * system.masses[1]) / dist2)  # absolute value of v-stable
abs_v_tangential = np.linalg.norm(v_rocket2_tangential)
r = (
    abs_v_stable / abs_v_tangential
)  # Factor between absolut value of v-stable and current tangantiel rocket-velocity
v_stable = (
    v2_planet1_shortest_dist[:, 0] + v_rocket2_tangential * r
)  # Calculating v_stable of injenction
v_injection = v_stable - v_rocket1_shortest_dist
v_boost = np.linalg.norm((v_rocket2[:, 0, idx2] - v_stable))
fuel_injection = calculate_needed_fuel(
    falcon_engine, dry_rocket_mass + (fuel_weight - fuel_boost1), v_boost, dt=0.00001
)  # Calculating needed fuel to perform injection


#################################################################
# #                 Rocket trajectory 3                       # #
#################################################################
## Simulating rocket trajectory after injection ##
(time_array, r_rocket3, v_rocket3, r_planets) = rocket_trajectory(
    time_of_least_distance2,
    r_rocket2[0, 0, idx2],
    r_rocket2[1, 0, idx2],
    v_stable[0],
    v_stable[1],
    total_flight_time=3 - time_of_least_distance2,
    time_step=10e-6,
)

#################################################################
# #  Plotting all three simulated rocket trajectories         # #
#################################################################
for i in range(7):
    plt.plot(
        r_planets1[0, i, :],
        r_planets1[1, i, :],
        linestyle="--",
        alpha=0.5,
        label=f"orbit{i}",
    )
plt.plot(
    r_rocket1[0, 0, 0 : int(idx1 + 1)],
    r_rocket1[1, 0, 0 : int(idx1 + 1)],
    label="Trj. aft. launch",
)
plt.plot(
    r_rocket2[0, 0, 0 : int(idx2 + 1)],
    r_rocket2[1, 0, 0 : int(idx2 + 1)],
    label="Trj. aft. corr. boost",
)
plt.plot(
    r_rocket3[0, 0, :],
    r_rocket3[1, 0, :],
    label="Trj. aft. injection",
)

plt.scatter(r_rocket2[0, 0, idx2], r_rocket2[1, 0, idx2], label="Min dist rocket", s=70)
plt.scatter(
    r_planets1[0, 1, idx2], r_planets1[1, 1, idx2], label="Min dist planet", s=70
)
plt.scatter(
    r_rocket1[0, 0, idx1], r_rocket1[1, 0, idx1], label="Rocket before boost", s=70
)
plt.scatter(0, 0, label="Sun", s=70)
plt.scatter(r_rocket1[0, 0, 0], r_rocket1[1, 0, 0], label="Rocket after launch", s=70)

plt.scatter(
    r_planets[0, 1, 0], r_planets[1, 1, 0], label="More accurate planet 1", s=70
)

plt.scatter(0.0593418, 0.0582803, label="Starting pos given unstable orbit2")

plt.xlabel("Au", fontsize=20)
plt.ylabel("Au", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=15)
plt.title("Rocket trajectory towards target planet", fontsize=20)
plt.show()


# ################################################################
# #   Interplanetary travel, with shortcut to unstable orbit    ##
# ################################################################
"""Trying pilot the rocket using the Interplanetary-instance (not enough time)"""
"""
flying = mission.begin_interplanetary_travel()
flying.coast_until_time(1.36614)
orient = flying.orient()
pos1 = orient[1]
plt.scatter(pos1[0], pos1[1], label="Actual pos1")
vel = orient[2]
print(orient)
print(orient[2])
print(vel[0], vel[1])
flying.boost(vel * 0.01)
flying.coast_until_time(1.36814)
orient2 = flying.orient()
pos2 = orient2[1]
plt.scatter(pos2[0], pos2[1], label="Actual pos 2")
vel2 = orient2[2]
flying.boost(-vel * 0.01)
flying.coast_until_time(time_of_least_distance)
orient3 = flying.orient()
pos3 = orient3[1]
plt.scatter(pos3[0], pos3[1], label="Actual pos 3")
flying.boost(np.array((boost_rocket_x * 100, boost_rocket_y * 100)))
flying.coast_until_time(time_of_least_distance2)
orient4 = flying.orient()
pos4 = orient4[1]
plt.scatter(pos4[0], pos4[1], label="Actual pos 4")
flying.look_in_direction_of_planet(1)
flying.take_picture()
orient_before_injection = flying.orient()
pos5 = orient_before_injection[1]
plt.scatter(pos5[0], pos5[1], label="Actual pos 5")
flying.boost(
    np.array(
        (v_rocket[0, 0, idx2][0] - v_stable[0], v_rocket[1, 0, idx2][0] - v_stable[0])
    )
)
flying.coast_until_time(2.99)
orient_after_injection = flying.orient()
pos6 = orient_after_injection[1]
plt.scatter(pos6[0], pos6[1], label="Actual pos 6")
plt.plot(
    (pos1[0], pos2[0], pos2[0], pos3[0], pos4[0], pos5[0], pos6[0]),
    (pos1[1], pos2[1], pos2[1], pos3[1], pos4[1], pos5[1], pos6[1]),
    label="Actual trajectory",
)
"""


def analyse_final_orbit_and_plot(
    type_of_orbit: str, orbit_time: float, time_step: float = 60
):
    """Function to create plot of final orbit around planet 1. Choose stable with
    input "stable" or unstable with input "unstable". Orbit_time is the period to analyse
    orbit in years. time_step is time_step of calculationg in seconds."""
    ######## Here we are using a shortcut #########
    if type_of_orbit == "unstable":
        code_unstable_orbit = 69696
        shortcut = SpaceMissionShortcuts(mission, [code_unstable_orbit])
        shortcut.place_spacecraft_in_unstable_orbit(
            time_of_least_distance2, 1
        )  # <----- Shortcut to unstable orbit ######## Shortcut here ########

    elif type_of_orbit == "stable":
        code_stable_orbit = 75980
        shortcut = SpaceMissionShortcuts(mission, [code_stable_orbit])
        shortcut.place_spacecraft_in_stable_orbit(
            time_of_least_distance2, 1000000, 0, 1
        )  # <----- Using shortcut to stable orbit

    land = mission.begin_landing_sequence()
    time = []
    x_pos_list = []
    y_pos_list = []
    x_vel_list = []
    y_vel_list = []
    orient = land.orient()
    pos = orient[1]
    vel = orient[2]
    time.append(0)
    x_pos_list.append(pos[0])
    y_pos_list.append(pos[1])
    x_vel_list.append(vel[0])
    y_vel_list.append(vel[1])
    t = 0
    revolutions = 0
    # while t < 5 * sec_per_year:
    while t < orbit_time * sec_per_year:
        t += time_step
        land.fall_until_time(t)
        orient = land.orient()
        pos = orient[1]
        vel = orient[2]
        time.append(t)
        x_pos_list.append(pos[0])
        y_pos_list.append(pos[1])
        x_vel_list.append(vel[0])
        y_vel_list.append(vel[1])
        if (y_pos_list[-1] > y_pos_list[0]) & (
            y_pos_list[-2] < y_pos_list[0]
        ):  # counting number of revolutions by checking if planet has crossed x-axis in this dt
            revolutions += 1
    time_array = np.array(time)
    pos_array = np.array((x_pos_list, y_pos_list))
    vel_array = np.array((x_vel_list, y_vel_list))
    abs_pos_array = np.sqrt(pos_array[0] ** 2 + pos_array[1] ** 2)
    abs_vel_array = np.sqrt(vel_array[0] ** 2 + vel_array[1] ** 2)

    ## Plotting results ##
    # Orbit
    plt.plot(pos_array[0], pos_array[1], label="Pos around planet 1 center")
    plt.scatter(0, 0, label="Planet 1 center")
    plt.xlabel("Meters [m]", fontsize=20)
    plt.ylabel("Meters [m]", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.axis("equal")
    plt.title(
        f"Untable orbit around planet 1 over 0.5 years. Number of revolutions: {revolutions}",
        fontsize=20,
    )
    plt.legend(fontsize=20)
    plt.show()

    # Distance
    plt.plot(
        time_array / sec_per_year,
        np.sqrt(pos_array[0] ** 2 + pos_array[1] ** 2),
        label="Distance from planet center",
    )

    plt.plot(
        time_array / sec_per_year,
        np.full(len(time_array), np.mean(abs_pos_array)),
        label="Mean distance from planet center",
    )
    plt.xlabel("Time [yr]", fontsize=20)
    plt.ylabel("Distance [m]]", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(
        f"Distance to planet 1 center over 0.5 year",
        fontsize=20,
    )
    plt.legend(fontsize=20)
    plt.show()

    # Velocity
    G = 6.6743 * (10 ** (-11))  # Gravitational constant
    plt.plot(
        time_array / sec_per_year,
        abs_vel_array,
        label="Absolute velocity",
    )
    plt.plot(
        time_array / sec_per_year,
        np.full(len(time_array), np.mean(abs_vel_array)),
        label="Mean absolute_velocity",
    )
    plt.ylabel("Velocity [m/s]", fontsize=20)
    plt.xlabel("Time [yr]", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(
        f"Absolute velocity of rocket and mean-abs velocity around planet 1 over 1 year",
        fontsize=20,
    )
    plt.legend(fontsize=20)
    plt.show()


analyse_final_orbit_and_plot("stable", 0.05)
