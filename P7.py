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
from scipy.optimize import curve_fit

# from P1B import Engine
# from P2 import simulate_orbits
# import h5py
# from part3 import generalized_launch_rocket
from PIL import Image

utils.check_for_newer_version()

np.random.seed(10)
seed = 57063
system = SolarSystem(seed)
mission = SpaceMission(seed)
Au = 149597870700  # Meters
SM = 1.9891 * 10 ** (30)  # Solar masses in kg
sec_per_year = 60 * 60 * 24 * 365
c_AU = 63239.7263  # Speed of light in Au/yr
c = const.c * (sec_per_year / Au)  # Speed of light in Au/yr
G_AU = 4 * (np.pi) ** 2  # Gravitational constant using AU
G = 6.6743 * (10 ** (-11))  # Gravitational constant
m_H2 = const.m_H2
k_B = const.k_B


lander_mass = mission.lander_mass  # 90.0 kg
lander_area = mission.lander_area  # 0.3 m^2
planet_mass = system.masses[1] * SM  # kg
planet_radius = system.radii[1] * 1000  # radius in meters
g0 = (G * planet_mass) / planet_radius**2  # gravitational acceleration at surface
mu = 

# def Landing(inital_time, inital_position, inital_velocity, simulation_time):
#     air_resistance =
#     gravity =
#     rotation =
#     return (final_time, final_position, final_velocity)


def landing_trajectory(
    initial_time,
    initial_pos,
    initial_vel,
    total_simulation_time,
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
        initial_time, initial_time + total_simulation_time, time_step
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

    x_pos_arr[0] = initial_pos[0]  # Setting initial x-position
    y_pos_arr[0] = initial_pos[1]  # Setting initial y-position
    x_vel_arr[0] = initial_vel[0]  # Setting initial x-velocity
    y_vel_arr[0] = initial_vel[1]  # Setting initial y-velocity

    r_rocket = np.array(
        [[x_pos_arr], [y_pos_arr]]
    )  # Empty pos-array for rocket position, to be filled with values from simulation
    v_rocket = np.array(
        [[x_vel_arr], [y_vel_arr]]
    )  # Empty vel-array for rocket position, to be filled with values from simulation

    a_planet = ((-G * planet_mass) * r_rocket[:, :, 0]) / (
        np.sqrt(np.sum(r_rocket[:, :, 0] ** 2))
    ) ** 3  # Sets initial acceleration from planet according to N.2 law
    acc_old = a_planet

    # Leapfrog-loop
    for i in range(0, len(time_array) - 1):
        rho = 1  # expression for rho
        v_terminal = np.sqrt(
            G
            * (2 * lander_mass * planet_mass * r_rocket[i])
            / (rho * lander_area * np.linalg.norm(r_rocket[i]) ** 3)
        )
        r_rocket[:, :, i + 1] = (
            r_rocket[:, :, i]
            + v_rocket[:, :, i] * time_step
            + (acc_old * time_step**2) / 2
        )  # Rocket pos at time i+1

        a_planet = (
            (-G * planet_mass)
            * r_rocket[:, :, i + 1]
            / (np.sqrt(np.sum(r_rocket[:, :, i + 1] ** 2)) ** 3)
        )  # Sets initial acceleration from sun according to N.2 law

        acc_new = a_planet  # Setting new acceleration to calculate velocity change
        v_rocket[:, :, i + 1] = (
            v_rocket[:, :, i] + (1 / 2) * (acc_old + acc_new) * time_step
        )  # Calculating velocty of rocket in timestep i+1
        acc_old = acc_new

    return time_array, r_rocket, v_rocket
