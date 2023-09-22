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
G = 4 * (np.pi) ** 2
planet_types = system.types  # ('rock', 'rock', 'gas', 'rock', 'rock', 'gas', 'rock')


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
        plt.scatter(0, 0, label="Sun", s=100)
        plt.scatter(CM[0], CM[1], label="CM")

        for i in range(number_of_planets):
            # ax.scatter(f[i] , np.sqrt(initial_positions[0][i]**2 + initial_positions[1][i]**2))
            plt.scatter(initial_positions[0][i], initial_positions[1][i], s=20)
            for j in range(N):
                r[i][j] = a[i] * (1 - e[i] ** 2) / (1 + e[i] * np.cos(t[j]))
            # ax.plot(t,r[i], label = f'{[i]}')
            plt.plot(r[i] * np.cos(t), r[i] * np.sin(t), label=f"{[i]}")
        plt.xlabel("AU")
        plt.ylabel("AU")
        plt.legend()
        plt.show()


# SolarSystem.analytical_plot()


def analytical_orbits(
    initial_pos_x,
    initial_pos_y,
    initial_vel_x,
    initial_vel_y,
    initial_theta,
    m,
    e,
    omega,
    dt=10e-9,
    T=1,
):
    """Calculate analytical orbits using the general solution of the two-body-problem.
    Takes initial position x/y, initial velocitie x/y, initial orbital angle (initial_theta), mass (m),
    eccentricites and aphelian angle (omega) of one planet. Calculates orbit and returns arrays of
    x_pos, y_pos, r_array, t_array, and theta_array."""
    t_array = np.arange(0, T, dt)
    N = len(t_array)
    theta_array = np.linspace(initial_theta, 2 * np.pi + initial_theta, N)
    r_array = np.zeros(len(t_array))
    r_0 = (initial_pos_x, initial_pos_y)
    v_0 = (initial_vel_x, initial_vel_y)
    h = np.abs(np.cross(r_0, v_0))  # calculate angular-momentum (constant)
    M = G * ((star_mass + m))
    p = (h**2) / M
    # for-loop loops over theta_array and calculates radius r for each step using general solution of two-body-problem
    for i in range(len(t_array)):
        f = theta_array[i] - omega + np.pi
        r_array[i] = p / (1 + e * np.cos(f))
    x_pos = (
        np.cos(theta_array) * r_array
    )  # convert theta/radius to cartesian x-coordinate
    y_pos = (
        np.sin(theta_array) * r_array
    )  # convert theta/radius to cartesian y-coordinate
    return x_pos, y_pos, r_array, t_array, theta_array


@njit
def simulate_orbits(
    initial_pos_x, initial_pos_y, initial_vel_x, initial_vel_y, dt=10e-9, T=1
):
    """Simulate orbits using newtons gravitational law. Takes initial position x/y,
    initial velocity x/y. Calculates orbit and returns arrays of
    x_pos, y_pos, t_array, and number of revolutions per earth-year."""
    # creating empty arrays
    t_array = np.arange(0, T, dt)
    x_pos = np.zeros(len(t_array))
    y_pos = np.zeros(len(t_array))
    x_vel = np.zeros(len(t_array))
    y_vel = np.zeros(len(t_array))
    # setting initial parameters
    x_pos[0] = initial_pos_x
    y_pos[0] = initial_pos_y
    x_vel[0] = initial_vel_x
    y_vel[0] = initial_vel_y
    gamma = -G * star_mass  # setting constant
    x_acc_old = (gamma * x_pos[0]) / (
        np.sqrt(x_pos[0] ** 2 + y_pos[0] ** 2)
    ) ** 3  # initial acceleration x-direction (Newtons-Gravitational law x-component)
    y_acc_old = (gamma * y_pos[0]) / (
        np.sqrt(x_pos[0] ** 2 + y_pos[0] ** 2)
    ) ** 3  # initial acceleration y-direction (Newtons-Gravitational law y-component)
    count_revolutions = 0
    round_timer = []
    crossing_idx = []
    # leapfrog method
    for i in range(1, len(t_array)):
        x_pos[i] = (
            x_pos[i - 1] + (x_vel[i - 1] * dt) + ((x_acc_old * dt**2) / 2)
        )  # updating x-pos
        y_pos[i] = (
            y_pos[i - 1] + (y_vel[i - 1] * dt) + ((y_acc_old * dt**2) / 2)
        )  # updating y-pos

        if (y_pos[i] > initial_pos_y) & (
            y_pos[i - 1] < initial_pos_y
        ):  # counting number of revolutions by checking if planet has crossed x-axis in this dt
            count_revolutions += 1
            round_timer.append(t_array[i])
            crossing_idx.append(i)

        x_acc_new = (gamma * x_pos[i - 1]) / (
            np.sqrt((x_pos[i - 1] ** 2) + (y_pos[i - 1] ** 2))
        ) ** 3  # setting new x-acceleration using the position of in the last iteration
        y_acc_new = (gamma * y_pos[i - 1]) / (
            np.sqrt(x_pos[i - 1] ** 2 + y_pos[i - 1] ** 2)
        ) ** 3  # setting new y-acceleration using the position of in the last iteration
        x_vel[i] = (
            x_vel[i - 1] + (1 / 2) * (x_acc_old + x_acc_new) * dt
        )  # updating x-velocity
        y_vel[i] = (
            y_vel[i - 1] + (1 / 2) * (y_acc_old + y_acc_new) * dt
        )  # updating y-velocity
        x_acc_old = x_acc_new  # setting old x-aceleration to new x-acceleration to prepare for next iteration
        y_acc_old = y_acc_new  # setting old y-aceleration to new y-acceleration to prepare for next iteration
        ## Calculating period
        round_timer_array = np.asarray(round_timer)
        delta_round_timer = np.diff(round_timer_array)
        period = delta_round_timer[0]
        # Calculating displacement
        crossings_r = []
        for c in crossing_idx:
            crossings_r.append(np.sqrt((x_pos[c] ** 2 + y_pos[c] ** 2)))
        crossing_r_array = np.asarray(crossings_r)
        initial_r = np.sqrt(initial_pos_x**2 + initial_pos_y**2)
        relative_displacement = crossing_r_array[0] / initial_r

    return x_pos, y_pos, t_array, count_revolutions, period, relative_displacement


def plot_orbits(T, dt):
    """A simple function that iterates over all planet indexes to plot the
    analytical and simulated orbit of each planet aswell as dots illustrating
    the initial positions of each planet, their relative readius-lengths and the type
    of planet (rock vs gas). Takes T (number of earth-years) and dt (timestep). Small dt increases accuracy.
    """
    # Calling simulate_orbit() for every planet and plotting result
    for i in range(len(initial_positions[0])):
        (
            sx_pos,
            sy_pos,
            t_array,
            count_revolutions,
            period,
            relative_displacement,
        ) = simulate_orbits(
            initial_positions[0][i],
            initial_positions[1][i],
            initial_velocities[0][i],
            initial_velocities[1][i],
            T=T,
            dt=dt,
        )
        plt.plot(
            sx_pos,
            sy_pos,
            label=f"Planet: {i}, revolutions: {count_revolutions}, period: {period:.2f}, displacement: {relative_displacement}",
        )
    # Calling analytic_orbits() for every planet and plotting result
    for i in range(len(initial_positions[0])):
        ax_pos, ay_pos, r_array, t_array, theta_array = analytical_orbits(
            initial_positions[0][i],
            initial_positions[1][i],
            initial_velocities[0][i],
            initial_velocities[1][i],
            initial_orbital_angles[i],
            masses[i],
            eccentricities[i],
            omega=aphelion_angles[i],
            T=T,
            dt=dt,
        )
        plt.plot(
            ax_pos,
            ay_pos,
            linestyle="dotted",
            color="black",
            alpha=0.5,
        )
    # Plotting initial_position of every planet, with relative radius represented in the size of the dot,
    # aswell as color red representing either gas-planet and blur representing rock-planet. Also annotates mass of each planet
    for i in range(len(initial_positions[0])):
        if planet_types[i] == "rock":
            color = "blue"
        else:
            color = "red"
        plt.scatter(
            initial_positions[0][i],
            initial_positions[1][i],
            color=f"{color}",
            s=radii[i] * 0.05,
        )  # scatterplot with size of dot = 0.05 * radius of planet

        plt.annotate(
            f"{masses[i]:.2E}", (initial_positions[0][i], initial_positions[1][i])
        )
    plt.xlabel("Au", fontsize=25)
    plt.ylabel("Au", fontsize=25)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.legend(loc="upper right", fontsize=15)
    plt.grid()
    plt.title(f"Analytical vs. simulated orbits (T={T}, dt = {dt})", fontsize=25)
    plt.show()


# plot_orbits(T=3, dt=10e-7)


# Task B
def test_kepler_laws(T, dt):
    """Function to test if simulated orbits obey Keplers-Laws."""
    # simulating the orbit of each planet by calling simulate_orbits().
    for p in range(len(masses)):
        (
            x_pos,
            y_pos,
            t_array,
            revolutions,
            period,
            relative_displacement,
        ) = simulate_orbits(
            initial_positions[0][p],
            initial_positions[1][p],
            initial_velocities[0][p],
            initial_velocities[1][p],
            T=T,
            dt=dt,
        )
        sm_axes = semi_major_axes[
            p
        ]  # storing the value of the Semi-Major-axis for later calculations
        n = (
            len(t_array) // revolutions
        )  # finding the number of timesteps n per revolution
        dn = n // 10  # choosing the delta_step to be n/10 - used in later calculation
        # calculating the area sweeped in area1, the positions from starting pos 0 to dn.
        area1 = 0  # created variable to update in loop
        distance1 = 0  # created variable to update in loop
        time1 = 0  # created variable to update in loop
        for i in range(0, dn):
            delta_r = (
                x_pos[i + 1] - x_pos[i],
                y_pos[i + 1] - y_pos[i],
            )  # delta_r is the distance between r[i+1] and r[i]. Here we approximate
            # the curvature of the circle to be a short straight line (because the delta_n is so small)
            r_vec = (x_pos[i], y_pos[i])
            area1 += (
                (1 / 2) * np.linalg.norm(r_vec) * np.linalg.norm((delta_r))
            )  # calculating the area of the triangle with base r_vec and height delta_r
            distance1 += np.linalg.norm(
                delta_r
            )  # the approximated distance traveled in this step is delta_r
            time1 += dt  # keeping track of time
        # Now doing the excact same thing for area2, from half revolution n/2 to n/2 + dn.
        # This is to compare the area_sweeped on tho opposite sides of the ellipse
        area2 = 0
        distance2 = 0
        time2 = 0
        for i in range(n // 2, (n // 2) + (dn)):
            delta_r = (
                x_pos[i + 1] - x_pos[i],
                y_pos[i + 1] - y_pos[i],
            )
            r_vec = (x_pos[i], y_pos[i])
            area2 += (1 / 2) * np.linalg.norm(r_vec) * np.linalg.norm((delta_r))
            distance2 += np.linalg.norm(delta_r)
            time2 += dt
        # Calculating mean velocity on the two opposing sides of the ellipse
        mean_vel1 = distance1 / time1
        mean_vel2 = distance2 / time2
        newton_improved_third_law = (
            (4 * np.pi**2) * (sm_axes**3) / (G * (star_mass + masses[p]))
        )  # calculating newtons improved version of Keplers third law-
        # Printing out the results found for each planet:
        print(f"----- Planet: {p} ----- ")
        print(f"Sim period: {period}")
        print(f"Number of periodd: {revolutions}")
        print(f"Relative r displacement per period: {relative_displacement}")
        print(f"Period squared:{period**2}")
        print(f"Newtons improved: {newton_improved_third_law}")
        print(f"SM-axes cubed:{sm_axes**3}")
        print("-------------------- ")
        print("\n")


# test_kepler_laws(3, 10e-7)

# for i in range(len(masses)):  #Checks which planet gives the most shift in the centre of mass.
#     CM = (1 / (star_mass + masses[i])) * ((masses[i] * np.array([initial_positions[0][i], initial_positions[1][i]])))
#     print(np.linalg.norm(CM)) #Gets planet nr.2, which we now will use in MovingTheSun function.


# Task C. Using planet 2
def moving_the_sun(T, dt):
    # T er lengden år
    # dt er tidssteg per år
    t_array = np.arange(0, T, dt)
    planet_mass = masses[2]
    planet_pos = np.array([initial_positions[0][2], initial_positions[1][2]])
    planet_vel = np.array([initial_velocities[0][2], initial_velocities[1][2]])
    star_pos = np.array([0, 0])  # Star starts in the origin
    star_vel = np.array([0, 0])  # Star starts with 0 velocity
    M = star_mass + planet_mass  # sum of masses
    CM = (1 / (M)) * (
        (planet_mass * planet_pos) + (star_mass * star_pos)
    )  # Finds CM, which we now will use as the origin, as it will be fixed to the origin.
    # print(planet_pos, star_pos, CM)
    # plt.scatter(star_pos[0], star_pos[1], label = 'Star')         #Checks tha the value for CM makes sense
    # plt.scatter(planet_pos[0], planet_pos[1], label = 'Planet')
    # plt.scatter(CM[0], CM[1], label = 'CM')
    # plt.legend()
    # plt.axis('equal')
    # plt.show()
    star_pos = -CM  # Star and planet are moved according to the CM.
    planet_pos = planet_pos - CM
    # plt.scatter(star_pos[0], star_pos[1], label = 'Star')         #Checks tha the new value for CM is at the origin
    # plt.scatter(planet_pos[0], planet_pos[1], label = 'Planet')
    # plt.scatter(CM_origin[0], CM_origin[1], label = 'CM')
    # plt.legend()
    # plt.axis('equal')
    # plt.show()
    N = int(T // dt)  # Definerer verdier til loopen
    star_pos_a = np.zeros((N, 2))
    star_pos_a[0] = star_pos
    planet_pos_a = np.zeros((N, 2))
    planet_pos_a[0] = planet_pos
    planet_vel_a = np.zeros((N, 2))
    star_vel_a = np.zeros((N, 2))
    planet_vel_a[0] = planet_vel
    star_vel_a[0] = star_vel
    r = planet_pos - star_pos
    a_star = (-G * planet_mass * r) / np.linalg.norm(r) ** 3  # Akselerasjon fra start
    a_planet = (-G * star_mass * r) / np.linalg.norm(r) ** 3
    for i in range(N - 1):  # Leapfrog integration using Newton
        CM = (1 / (M)) * ((planet_mass * planet_pos_a[i]) + (star_mass * star_pos_a[i]))
        star_pos_a[i + 1] = (
            star_pos_a[i] + star_vel_a[i] * dt + 0.5 * a_star * dt**2 - CM
        )  # Trekker fra bevegelsen til CM slik at den ikke beveger seg
        planet_pos_a[i + 1] = (
            planet_pos_a[i] + planet_vel_a[i] * dt + 0.5 * a_planet * dt**2 - CM
        )
        r = planet_pos_a[i + 1] - star_pos_a[i + 1]
        a_star_next = (-G * planet_mass * r) / np.linalg.norm(r) ** 3
        a_planet_next = (-G * star_mass * r) / np.linalg.norm(r) ** 3
        star_vel_a[i + 1] = star_vel_a[i] + 0.5 * (a_star + a_star_next) * dt
        planet_vel_a[i + 1] = planet_vel_a[i] + 0.5 * (a_planet + a_planet_next) * dt
        a_star = a_star_next
        a_planet = a_planet_next
    # plt.plot(star_pos_a[:, 0], star_pos_a[:, 1], label="Star")
    # plt.plot(planet_pos_a[:, 0], planet_pos_a[:, 1], label="Planet")
    # plt.legend()
    # plt.show()
    mu = (planet_mass * star_mass) / (planet_mass + star_mass)
    star_pos_x = star_pos_a[:, 0]
    star_pos_y = star_pos_a[:, 1]
    planet_pos_x = planet_pos_a[:, 0]
    planet_pos_y = planet_pos_a[:, 1]
    star_vel_x = star_vel_a[:, 0]
    star_vel_y = star_vel_a[:, 1]
    planet_vel_x = planet_vel_a[:, 0]
    planet_vel_y = planet_vel_a[:, 1]
    E_planet = (
        (1 / 2) * (mu) * (np.sqrt(planet_vel_x**2 + planet_vel_y**2)) ** 2
    ) - (G * M * mu) / (np.sqrt(planet_pos_x**2 + planet_pos_y**2))
    E_star = ((1 / 2) * (mu) * (np.sqrt(star_vel_x**2 + star_vel_y**2)) ** 2) - (
        G * M * mu
    ) / (np.sqrt(star_pos_x**2 + star_pos_y**2))
    # E_cm = E_planet + E_star
    E_cm = (
        (1 / 2)
        * (mu)
        * (
            np.sqrt(planet_vel_x**2 + planet_vel_y**2)
            + np.sqrt(star_vel_x**2 + star_vel_y**2)
        )
        ** 2
    ) - (G * M * mu) / (
        np.sqrt(planet_pos_x**2 + planet_pos_y**2)
        + np.sqrt(star_pos_x**2 + star_pos_y**2)
    )
    Ek_cm = (
        (1 / 2)
        * (mu)
        * (
            np.sqrt(planet_vel_x**2 + planet_vel_y**2)
            + np.sqrt(star_vel_x**2 + star_vel_y**2)
        )
        ** 2
    )
    Eu_cm = -(G * M * mu) / (
        np.sqrt(planet_pos_x**2 + planet_pos_y**2)
        + np.sqrt(star_pos_x**2 + star_pos_y**2)
    )
    # plt.plot(t_array[:-1], E_planet, label="E planet")
    # plt.plot(t_array[:-1], E_star, label="E star")
    plt.plot(t_array[:-1], E_cm, label="E CM")
    plt.plot(t_array[:-1], Ek_cm, label="Kineting E CM")
    plt.plot(t_array[:-1], Eu_cm, label="Potensial E CM")
    plt.xlabel("t", fontsize=25)
    plt.ylabel("J", fontsize=25)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.legend(loc="upper right", fontsize=15)
    plt.grid()
    plt.title(
        f"Total-, kinetic- and potensial energy of CM (T={T}, dt={dt})", fontsize=25
    )
    plt.show()


moving_the_sun(T=1, dt=10e-6)
