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
G = 6.6743 * 10 ** (-11)
SM = 1.9891 * 10 ** (30)


class SolarSystem:
    def analytical_plot():
        N = 1000
        theta = 2 * np.pi
        t = np.linspace(0, theta, N)
        aX = semi_major_axes
        a = np.zeros(number_of_planets)
        e = eccentricities
        f = aphelion_angles + np.pi
        r = np.zeros((number_of_planets, N))

        M = np.sum(masses + star_mass)
        CM = (
            np.array(
                [
                    np.sum(masses * initial_positions[0]),
                    np.sum(masses * initial_positions[1]),
                ]
            )
            / M
        )
        # CMr = np.sqrt(sum(CM**2))
        # CMt = np.arccos(CM[0] / CMr)
        for i in range(number_of_planets):
            mu = (masses[i] * star_mass) / (masses[i] + star_mass)
            a[i] = mu * aX[i] / masses[i]

        # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # ax.scatter(0,0, label = 'Sun')
        # ax.scatter(CMt, CMr, label = 'CM')
        plt.scatter(0, 0, label="Sun", s = 100)
        plt.scatter(CM[0], CM[1], label="CM")

        for i in range(number_of_planets):
            # ax.scatter(f[i] , np.sqrt(initial_positions[0][i]**2 + initial_positions[1][i]**2))
            plt.scatter(initial_positions[0][i], initial_positions[1][i], s = 20)
            for j in range(N):
                r[i][j] = a[i] * (1 - e[i] ** 2) / (1 + e[i] * np.cos(t[j]))
            # ax.plot(t,r[i], label = f'{[i]}')
            plt.plot(r[i] * np.cos(t), r[i] * np.sin(t), label=f"{[i]}")
        plt.xlabel("AU")
        plt.ylabel("AU")
        plt.legend()
        plt.show()


SolarSystem.analytical_plot()


@njit
def simulate_orbit(
    initial_pos_x,
    initial_pos_y,
    initial_vel_x,
    initial_vel_y,
    initial_angle,
    m,
    dt=1,
    T=1000,
):
    t_array = np.arange(0, T, dt)
    # Kartesisk
    x_pos = np.zeros(len(t_array))
    y_pos = np.zeros(len(t_array))
    x_vel = np.zeros(len(t_array))
    y_vel = np.zeros(len(t_array))
    # x_acc = np.zeros(len(t_array))
    # y_acc = np.zeros(len(t_array))
    # Sfærisk
    # r_array = np.zeros(len(t_array))
    # v_array = np.zeros(len(t_array))
    # theta_array = np.zeros(len(t_array))
    # delta_theta_array = np.zeros(len(t_array))
    # dd_theta_array = np.zeros(len(t_array))

    # theta_acc = np.zeros(len(t_array))
    # f_array = np.zeros(len(t_array))

    x_pos[0] = initial_pos_x
    y_pos[0] = initial_pos_y
    x_vel[0] = initial_vel_x
    y_pos[0] = initial_vel_y
    # r_array[0] = np.sqrt(x_pos[0] ** 2 + y_pos[0] ** 2)
    # v_array[0] = np.sqrt(x_vel[0] ** 2 + y_vel[0] ** 2)
    # theta_array[0] = initial_angle
    # delta_theta_array = np.sqrt(x_vel[0] ** 2 + y_vel[0] ** 2) / np.sqrt(
    # x_pos[0] ** 2 + y_pos[0] ** 2
    # )
    # f_array[0] = (G * star_mass * m) / (r_array[0] ** 2)
    x_acc_old = (G * star_mass) / (x_pos[0] ** 2)
    y_acc_old = (G * star_mass) / (y_pos[0] ** 2)
    # dd_theta_array[0] = np.sqrt(x_acc[0]**2 + y_acc[0]**2)/(m*np.sqrt(x_pos[0]**2 + y_pos[0]**2))

    # leapfrog method
    for i in range(1, len(t_array)):
        x_pos[i] = x_pos[i - 1] + x_vel[i - 1] * dt + 1 / 2 * x_acc_old * dt**2
        y_pos[i] = y_pos[i - 1] + y_vel[i - 1] * dt + 1 / 2 * y_acc_old * dt**2
        x_acc_new = (G * star_mass) / (x_pos[i - 1] ** 2)
        y_acc_new = (G * star_mass) / (x_pos[i - 1] ** 2)
        x_vel[i] = x_vel[i - 1] + x_vel[i - 1] + 1 / 2 * (x_acc_old + x_acc_new)
        y_vel[i] = y_vel[i - 1] + y_vel[i - 1] + 1 / 2 * (y_acc_old + y_acc_new)
        x_acc_old = x_acc_new
        y_acc_old = y_acc_new
    return x_pos, y_pos, t_array


x_pos, y_pos, t_array = simulate_orbit(
    initial_positions[0][0],
    initial_positions[1][0],
    initial_velocities[0][0],
    initial_velocities[1][0],
    aphelion_angles[0],
    masses[0] * SM,
)
# print(x_pos)
plt.plot(x_pos, y_pos)
plt.show()
