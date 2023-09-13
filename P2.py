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

utils.check_for_newer_version()

seed = 57063
system = SolarSystem(seed)
mission = SpaceMission(seed)

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
G = 6.6743 * 10 ** (-11)


class SolarSystem:
    def analytical_plot():
        T = 1000
        N = 10000
        t = np.linspace(0, T, N)
        rx = np.zeros(N)
        ry = np.zeros(N)
        CM = np.zeros(N)

        rx[0] = initial_positions[0]
        ry[0] = initial_positions[1]
        for i in range(N):
            rm = np.array([np.sum(masses * rx), np.sum(masses * ry)])
            CM[i] = rm / (
                star_mass + np.sum(masses)
            )  # sum of m_i * r_i  / M.  Sola er i origo
            # Leapfrog

        plt.legend()
        plt.show()


SolarSystem.analytical_plot()


def simulate_orbit(
    initial_pos_x,
    initial_pos_y,
    initial_vel_x,
    initial_vel_y,
    initial_angle,
    m,
    A,
    omega,
    dt=0.1,
    T=10000,
):
    t_array = np.arange(0, T, dt)
    x_pos = np.zeros(len(t_array))
    y_pos = np.zeros(len(t_array))
    x_vel = np.zeros(len(t_array))
    y_vel = np.zeros(len(t_array))
    x_acc = np.zeros(len(t_array))
    y_acc = np.zeros(len(t_array))
    r_array = np.zeros(len(t_array))
    v_array = np.zeros(len(t_array))
    theta_array = np.zeros(len(t_array))
    delta_theta_array = np.zeros(len(t_array))
    # theta_acc = np.zeros(len(t_array))
    f_array = np.zeros(len(t_array))

    x_pos[0] = initial_pos_x
    y_pos[0] = initial_pos_y
    x_vel[0] = initial_vel_x
    y_pos[0] = initial_vel_y
    r_array[0] = np.sqrt(x_pos[0] ** 2 + y_pos[0] ** 2)
    v_array[0] = np.sqrt(x_vel[0] ** 2 + y_vel[0] ** 2)
    theta_array[0] = initial_angle
    delta_theta_array = np.sqrt(x_vel[0] ** 2 + y_vel[0] ** 2) / np.sqrt(
        x_pos[0] ** 2 + y_pos[0] ** 2
    )
    f_array[0] = (G * star_mass * m) / (r_array[0] ** 2)
    x_acc[0] = f_array[0] * np.cos(theta_array[0]) / m
    y_acc[0] = f_array[0] * np.sin(theta_array[0]) / m
    # theta_acc[0] = f_array[0]/m

    # leapfrog method
    for i in range(1, len(t_array)):
        x_pos[i] = x_pos[i - 1] + x_vel[i - 1] * dt + 1 / 2 * x_acc[i - 1] * dt**2
        y_pos[i] = y_pos[i - 1] + y_vel[i - 1] * dt + 1 / 2 * y_acc[i - 1] * dt**2
        theta_array[i] = theta_array[i - 1] + delta_theta_array[0] * dt
        f_array[i] = (G * star_mass * m) / (np.sqrt(x_pos[i] ** 2 + y_pos[i] ** 2)) ** 2
        x_acc[i] = f_array[i] * np.cos(theta_array[i]) / m
        y_acc[i] = f_array[i] * np.sin(theta_array[i]) / m
        x_vel[i] = x_vel[i - 1] + x_vel[i - 1] + 1 / 2 * (x_acc[i - 1] + x_acc[i])
        y_vel[i] = y_vel[i - 1] + y_vel[i - 1] + 1 / 2 * (y_acc[i - 1] + y_acc[i])
        r_array[i] = np.sqrt(x_pos[i] ** 2 + y_pos[i] ** 2)

    return x_pos, y_pos, x_vel, y_vel
