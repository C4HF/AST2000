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
import multiprocessing
import time
import math
# from P1B import Engine
# from P2 import simulate_orbits
# import h5py

utils.check_for_newer_version()

seed = 57063
system = SolarSystem(seed)
mission = SpaceMission(seed)
Au = 149597870700  # Meters
SM = 1.9891 * 10 ** (30)  # Solar masses in kg
star_mass = system.star_mass  # 0.25361200295275615
star_radius = system.star_radius  # 239265.2554658649
number_of_planets = system.number_of_planets  # 7

Sun_doppler_shift = mission.star_doppler_shifts_at_sun #(-0.020473606152657177, 0.01606904976188539)
Star_direction_angles = mission.star_direction_angles #(213.2764103110655, 149.62013634196333)

c = const.c
lamba_0 =  656.3    #nanometers
v_r_sol = c * (np.array(Sun_doppler_shift) / lamba_0)   #nanometers / nanometers [-9352.17539636  7340.2101567 ]

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


# distances = mission.measure_distances()


def create_orbits(T, dt):
    """Simulates orbits, packed like: x_pos, y_pos, x_vel, y_vel, t_array, count_revolutions,period, relative_displacement,"""
    for i in range(7):
        globals()[f"orbit_{i}"] = simulate_orbits(
            initial_positions[0][i],
            initial_positions[1][i],
            initial_velocities[0][i],
            initial_velocities[1][i],
            T=T,
            dt=dt,
        )


create_orbits(T=3, dt=10e-7)


def generalized_launch_rocket(
    engine,
    fuel_weight,
    target_vertical_velocity,
    launch_theta,
    launch_phi,
    launch_time,
    dt=1,
):
    """Funksjonen tar inn instans av engine, start-fuel-vekt, ønsket hastighet, vinkel-posisjon mellom nordpol/sorpol (launch_theta),
    vinkelposisjon langs ekvator på planeten med vinkel null langs x-aksen (launch-phi) og oppskytningstidspunkt T i jordår fra 0-3.
    Regner deretter ut akselereasjon med hensyn på gravitasjon og regner ut hastighet og posisjon.
    Funksjonen returnerer høyde over jordoverflaten, vertikal-hastighet, total-tid, resterende drivstoffvekt
    samt xy-posisjon og xy-hastighet i forhold til stjernen i solsystemet vårt."""
    thrust = engine.thrust
    total_fuel_constant = engine.total_fuel_constant
    sec_per_year = 60 * 60 * 24 * 365
    homeplanet_radius_Au = homeplanet_radius / Au  # Converting radius in meters to Au
    rotational_velocity = (
        np.abs(2 * np.pi * (homeplanet_radius_Au) * np.cos((np.pi / 2) - launch_theta))
    ) / (home_planet_rotational_period / 365)

    # finding idx of launch time
    for i, t in enumerate(orbit_0[4]):
        if math.isclose(t, launch_time):
            idx = i
            break
        else:
            continue
    solar_x_pos = orbit_0[0][idx] + (
        (
            (homeplanet_radius_Au)
            * np.cos((np.pi / 2) - launch_theta)
            * np.cos(launch_phi)
        )
    )  # Au
    solar_y_pos = orbit_0[1][idx] + (
        (
            (homeplanet_radius_Au)
            * np.cos((np.pi / 2) - launch_theta)
            * np.sin(launch_phi)
        )
    )  # Au
    solar_x_vel = orbit_0[2][idx] + rotational_velocity * (-np.sin(launch_phi))  # Au/yr
    solar_y_vel = orbit_0[3][idx] + rotational_velocity * np.cos(launch_phi)  # Au/yr

    altitude = 0  # m
    vertical_velocity = 0  # m/s
    total_time = 0  # s

    # While loop Euler-method with units kg, meters and seconds
    while vertical_velocity < target_vertical_velocity:
        wet_rocket_mass = dry_rocket_mass + fuel_weight
        F_g = (G * homeplanet_mass * wet_rocket_mass) / (
            (homeplanet_radius) + altitude
        ) ** 2  # The gravitational force
        rocket_thrust_gravitation_diff = thrust - F_g  # netto-kraft
        vertical_velocity += (
            rocket_thrust_gravitation_diff / wet_rocket_mass
        ) * dt  # m/s
        altitude += vertical_velocity * dt  # m
        # solar_x_pos += vertical_velocity * np.cos(launch_phi) * dt / Au  # Au
        # solar_y_pos += vertical_velocity * np.sin(launch_phi) * dt / Au  # Au
        fuel_weight -= total_fuel_constant * dt  # kg
        total_time += dt  # s

        if fuel_weight <= 0:
            break
        elif total_time > 1800:
            break
        elif altitude < 0:
            break
    solar_x_vel += vertical_velocity * np.cos(launch_phi) * (sec_per_year / Au)
    solar_y_vel += vertical_velocity * np.sin(launch_phi) * (sec_per_year / Au)
    solar_x_pos += altitude * np.cos(launch_phi) / Au
    solar_y_pos += altitude * np.sin(launch_phi) / Au
    return (
        altitude,
        vertical_velocity,
        total_time,
        fuel_weight,
        solar_x_pos,
        solar_y_pos,
        solar_x_vel,
        solar_y_vel,
    )


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
    # plt.plot(circle_2[0], circle_2[1], label="circle homeplanet")
    # plt.plot(circle_3[0], circle_3[1], label="circle 3")
    plt.legend()
    plt.show()


distance_list = [0.2, 0.1, 0.4, 0.2, 0.1, 0.7, 0.9, 0.8]
spacecraft_triliteration(2, distance_list)

falcon_engine = Engine(
    N=2 * 10**4, L=3.775 * 10e-8, n_A=1, T=3300, t_c=10e-11, dt=10e-14
)
