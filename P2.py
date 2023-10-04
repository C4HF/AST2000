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
import h5py
from pathlib import Path


utils.check_for_newer_version()

np.random.seed(10)
seed = 57063
# seed = 57023
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

    return (
        x_pos,
        y_pos,
        x_vel,
        y_vel,
        t_array,
        count_revolutions,
        period,
        relative_displacement,
    )


def save_orbits_to_file(T, dt):
    """Function to save simulated orbits in -file."""
    for i in range(len(initial_positions[0])):
        (
            x_pos,
            y_pos,
            x_vel,
            y_vel,
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
        h5f = h5py.File(f"orbit{i}.h5", "w")
        h5f.create_dataset("dataset_1", data=[t_array, x_pos, y_pos, x_vel, y_vel])
        h5f.close()


# save_orbits_to_file(T=3, dt=10e-6)


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
            x_vel,
            y_vel,
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
    plt.xlabel("Au", fontsize=10)
    plt.ylabel("Au", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc="upper right", fontsize=8)
    plt.grid()
    plt.title(f"Analytical vs. simulated orbits (T={T}, dt = {dt})", fontsize=25)
    plt.show()


# plot_orbits(T=3, dt=10e-6)


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
        print(f"Mean velocity 1: {mean_vel1}")
        print(f"Mean velocity 2: {mean_vel2}")
        print("-------------------- ")
        print("\n")


# test_kepler_laws(3, 10e-7)


# Task C. Using planet 2
@njit
def moving_the_sun(T, dt):
    """Function simulates new orbits for sun and planet with reference to
    center of mass usin leapfrog with Newtons law of gravitation. With this orbits
    it also calculates the potential and kinetic energy. Returns arrays with results."""
    # T is the amount of time in years
    # dt is the timestep per year
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
    star_pos = -CM  # Star and planet are moved according to the CM.
    planet_pos = planet_pos - CM
    N = len(t_array)
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
    ## Plot star pos and planet pos ##
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
    # E_cm = E_planet + E_star. Using analytic solution showed in the pdf.
    E_cm = (1 / 2) * (mu) * (  # Energy of the center of mass
        np.sqrt(planet_vel_x**2 + planet_vel_y**2)
        + np.sqrt(star_vel_x**2 + star_vel_y**2)
    ) ** 2 - (G * M * mu) / (
        np.sqrt(planet_pos_x**2 + planet_pos_y**2)
        + np.sqrt(star_pos_x**2 + star_pos_y**2)
    )
    Ek_cm = (  # Kinetic energy of the center of mass
        (1 / 2)
        * (mu)
        * (
            np.sqrt(planet_vel_x**2 + planet_vel_y**2)
            + np.sqrt(star_vel_x**2 + star_vel_y**2)
        )
        ** 2
    )
    Eu_cm = -(G * M * mu) / (  # Potential energy of the center of mass
        np.sqrt(planet_pos_x**2 + planet_pos_y**2)
        + np.sqrt(star_pos_x**2 + star_pos_y**2)
    )
    return star_pos_a, star_vel_a, planet_pos_a, E_cm, Ek_cm, Eu_cm, t_array


def plot_energy(T, dt):
    """Simple function that plots the energy of the new orbits."""
    star_pos_a, star_vel_a, planet_pos_a, E_cm, Ek_cm, Eu_cm, t_array = moving_the_sun(
        T=T, dt=dt
    )
    plt.plot(t_array, E_cm, label="Total E CM")
    plt.plot(t_array, Ek_cm, label="Kineting E CM")
    plt.plot(t_array, Eu_cm, label="Potensial E CM")
    plt.xlabel("År", fontsize=25)
    plt.ylabel("Energi", fontsize=25)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.legend(loc="upper right", fontsize=15)
    plt.grid()
    plt.title(
        f"Total-, kinetic- and potensial energy of CM (T={T}, dt={dt})", fontsize=25
    )
    plt.show()


# plot_energy(T=1, dt=10e-8)


def radial_velocity(T, dt):
    """Function creates a plot of the stars radial-velocity as seen from outside our solar-system,
    in the x/y-plane. Adds Gaussian-noise to simulate signal-disturbances."""
    star_pos_a, star_vel_a, planet_pos_a, E_cm, Ek_cm, Eu_cm, t_array = moving_the_sun(
        T=T, dt=dt
    )
    t = t_array  # Time array for plotting
    vx = star_vel_a[:, 0]  # Gathers the x-values for the velocities of the sun.
    mean = 0  # Defines values for the noise. Mean = 0 due to the noise being equal on both sides of the curve.
    std = 0.2 * max(vx)  # Std = 1/5 of the maximum value of the velocity
    GaussianNoise = np.random.normal(
        mean, std, size=(len(t))
    )  # Normal distribution using values defined above
    vx_GaussianNoise = vx + GaussianNoise  # Adding the noise to the velocity
    plt.plot(t, vx_GaussianNoise, label="v med støy")
    plt.plot(t, vx, label="Original v")

    plt.legend(fontsize=20)
    plt.xlabel("t (år)", fontsize=20)
    plt.ylabel("v (AU / år)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()


# radial_velocity(T=1, dt=10e-7)


def calculate_exoplanet_mass(m_s, v_r, P, i=np.pi / 2):
    """Simple function that calculates the estimated mass of a planet using mass of star (m_s),
    radial_velocity (v_r) and period (P). Returns mass of planet (mp)."""
    mp = (m_s ** (2 / 3) * v_r * P ** (1 / 3)) / (
        (2 * np.pi * G) ** (1 / 3) * np.sin(i)
    )
    return mp


# print(calculate_exoplanet_mass(m_s=3.660552991847526, v_r=0.000004, P=8.6))
# print((3.097971410491892e-6) / (3.290e-06))


def light_curve(T=0.000003, dt=10e-11):
    """Function creates a plot showing the relative changement in flux when a star
    is passed by a planet."""
    dia_planet = 2 * (radii[2] / Au)
    dia_star = 2 * (star_radius / Au)
    mean_vel = (5.08333400535115 + 5.37349607140599) / 2
    dt_t = dia_planet / mean_vel
    dt_lf = dia_star / mean_vel
    area_planet = np.pi * radii[2] ** 2
    area_star = np.pi * star_radius**2
    t_maxflux_array1 = np.arange(0, (T / 3), dt)
    t_transition_array = np.arange((T / 3) + dt_t, (T / 3) + dt_t + dt_lf, dt)
    t_maxflux_array2 = np.arange((T / 3) + dt_t + dt_lf + dt_t, T, dt)
    t_array = np.concatenate(
        (t_maxflux_array1, t_transition_array, t_maxflux_array2), axis=None
    )
    max_flux_array1 = np.zeros(len(t_maxflux_array1))
    max_flux_array1.fill(1)
    max_flux_array2 = np.zeros(len(t_maxflux_array2))
    max_flux_array2.fill(1)
    low_flux_array = np.zeros(len(t_transition_array))
    low_flux_array.fill(1 * (1 - (area_planet / area_star)))
    flux_array = np.concatenate(
        (max_flux_array1, low_flux_array, max_flux_array2), axis=None
    )
    mean = 0  # Defines values for the noise. Mean = 0 due to the noise being equal on both sides of the curve.
    std = 10 ** (-4)  # Std = 10^4
    gaussian_noise = np.random.normal(
        mean, std, size=(len(flux_array))
    )  # Normal distribution using values defined above
    hours_pr_year = 24 * 365
    plt.plot(
        t_array * hours_pr_year,
        flux_array + gaussian_noise,
        color="orange",
        label="Flux",
    )
    plt.xlabel("Time [hours]", fontsize=20)
    plt.ylabel("Relative flux [F_r]", fontsize=20)
    plt.title("Relative flux passing planet", fontsize=20)
    plt.minorticks_on()
    plt.grid(which="major", color="#DDDDDD", linewidth=0.8)
    plt.grid(which="minor", color="#EEEEEE", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()


# light_curve()


def calculate_exoplanet_density():
    """Function calculates and return explanet radius (r) and mean density (rho)"""
    v_r = 0.000004
    m_p = calculate_exoplanet_mass(m_s=3.660552991847526, v_r=0.000004, P=8.6)
    m_s = 3.660552991847526
    v_p = (v_r * m_s) / (m_p)
    delta_t = 0.8 / (24 * 365)
    r = (v_r + v_p) * (delta_t / 2) * (Au / 1000)
    rho = (m_p * SM) / (4 * np.pi * r**3 / 3)
    print(f"Planet radius [Km]: {r}")
    print(f"Planet mean-density [kg/km**3]: {rho}")


# calculate_exoplanet_density()
