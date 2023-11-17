########## Ikke kodemal #############################
import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem

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
m_O2 = 2.6566962 * 10e-26  # mass of molecule in kg  <---- found in atmosphere
m_H2O = 2.989 * 10e-26  # mass of molecule in kg <---- found in atmosphere
m_CO2 = 2.403 * 10e-26  # mass of molecule in kg
m_CH4 = 4.435 * 10e-26  # mass of molecule in kg <---- found in atmosphere
m_CO = 4.012 * 10e-26  # mass of molecule in kg
m_N2O = 7.819 * 10e-26  # mass of molecule in kg


lander_mass = mission.lander_mass  # 90.0 kg
lander_area = mission.lander_area  # 0.3 m^2
parachute_area = 24.5  # m^2
planet_mass = system.masses[1] * SM  # kg
planet_radius = system.radii[1] * 1000  # radius in meters
g0 = (G * planet_mass) / planet_radius**2  # gravitational acceleration at surface
mu = 22 / 1000  # mean molecular weight
T0 = 296.9  # surface temperature in kelvin found in part 3
K = (2 * g0 * mu * m_H2) / (k_B * T0)
rho0 = system.atmospheric_densities[1]  # density of atmosphere at surface
P = system.rotational_periods[1] * (
    60 * 60 * 24
)  # rotational period of planet in seconds
Cd = 1  # drag coeffisient of lander
rotation_matrix = np.array([[0, -1], [1, 0]])


def landing_trajectory(
    initial_time,
    initial_pos,
    initial_vel,
    total_simulation_time,
    time_step,
):
    """Function to calculate lander trajectory while falling through the atmosphere of planet 1.
    Takes initial time, position, velocity, simulation time and the timestep for the
    simulation (a smaller timestep increases accuracy). The function uses a leapfrog-loop to calculate
    acceleration from gravity and atmospheric air-resistance. Returns the trajectory of the lander including time,
    velocity and position.
    """
    lander_area = mission.lander_area  # 0.3 m^2
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
    v_terminal = np.zeros(len(time_array))

    x_pos_arr[0] = initial_pos[0]  # Setting initial x-position
    y_pos_arr[0] = initial_pos[1]  # Setting initial y-position
    x_vel_arr[0] = initial_vel[0]  # Setting initial x-velocity
    y_vel_arr[0] = initial_vel[1]  # Setting initial y-velocity

    r_lander = np.array(
        [[x_pos_arr], [y_pos_arr]]
    )  # Empty pos-array for rocket position, to be filled with values from simulation
    v_lander = np.array(
        [[x_vel_arr], [y_vel_arr]]
    )  # Empty vel-array for rocket position, to be filled with values from simulation

    Fd = np.zeros_like(v_lander)
    rho = rho0 * np.e ** (
        -K * np.linalg.norm(r_lander[:, :, 0]) - planet_radius
    )  # expression for rho (simplified model using isotherm atmosphere)
    v_terminal[0] = np.sqrt(
        G
        * (2 * lander_mass * planet_mass)
        / (rho * lander_area * (np.linalg.norm(r_lander[:, :, 0])) ** 2)
    )  # Calculating initial terminal velocity

    wind_magnitude = (
        2 * np.pi * np.linalg.norm(r_lander[:, :, 0]) / P
    )  # the wind from planet rotation at radius R from center
    wind = wind_magnitude * np.dot(
        rotation_matrix, r_lander[:, :, 0] / np.linalg.norm(r_lander[:, :, 0])
    )  # calculating the atmospheric wind

    v_drag = -(v_lander[:, :, 0] + wind)  # calculating the total drag-veloctiy
    # print(
    #     f"rho: {rho}, Cd: {Cd}, lander_area: {lander_area}, abs vdrag: {np.linalg.norm(v_drag)},vdrag vector {v_drag}"
    # )
    Fd[:, :, 0] = (
        1 / 2 * rho * Cd * lander_area * np.linalg.norm(v_drag) * v_drag
    )  # calculating the force of air-resistance
    a_drag = Fd[:, :, 0] / lander_mass  # acceleration from drag
    a_planet = (
        (-G * planet_mass)
        * r_lander[:, :, 0]
        / (np.sqrt(np.sum(r_lander[:, :, 0] ** 2)) ** 3)
    )  # Sets initial acceleration from sun according to N.2 law

    a_planet = ((-G * planet_mass) * r_lander[:, :, 0]) / (
        np.linalg.norm(r_lander[:, :, 0]) ** 3
    )  # Sets initial acceleration from planet according to N.2 law
    acc_old = a_planet + a_drag  # total acceleration

    # Leapfrog-loop
    for i in range(0, len(time_array) - 1):
        # print(f"Calculating {i}/{len(time_array) - 1}")
        if np.linalg.norm(r_lander[:, :, i]) <= planet_radius + 500:
            lander_area = lander_area + parachute_area

        if np.linalg.norm(r_lander[:, :, i]) <= planet_radius or np.linalg.norm(
            r_lander[:, :, i]
        ) > 2 * np.linalg.norm(r_lander[:, :, 0]):
            break

        r_lander[:, :, i + 1] = (
            r_lander[:, :, i]
            + v_lander[:, :, i] * time_step
            + (acc_old * time_step**2) / 2
        )  # Rocket pos at time i+1

        rho = rho0 * np.exp(
            -K * np.linalg.norm(r_lander[:, :, i] - planet_radius)
        )  # expression for rho (simplified model using isotherm atmosphere)
        wind_magnitude = (
            2 * np.pi * np.linalg.norm(r_lander[:, :, i + 1]) / P
        )  # the wind from planet rotation at radius R from center
        wind = wind_magnitude * np.dot(
            rotation_matrix, r_lander[:, :, i] / np.linalg.norm(r_lander[:, :, i])
        )  # calulating atmospheric wind
        v_terminal[i + 1] = np.sqrt(
            G
            * (2 * lander_mass * planet_mass)
            / (rho * lander_area * (np.linalg.norm(r_lander[:, :, i])) ** 2)
        )  # calculating terminal-velocity
        v_drag = -(v_lander[:, :, i] + wind)  # calculating drag velocity

        Fd[:, :, i] = (
            1 / 2 * rho * Cd * lander_area * np.linalg.norm(v_drag) * v_drag
        )  # air resistance
        a_drag = Fd[:, :, i] / lander_mass  # acceleration from air resistance
        a_planet = ((-G * planet_mass) * r_lander[:, :, i]) / (
            np.linalg.norm(r_lander[:, :, i]) ** 3
        )  # Sets initial acceleration from planet according to N.2 law # Sets initial acceleration from sun according to N.2 law
        acc_new = (
            a_planet + a_drag
        )  # Setting new acceleration to calculate velocity change
        v_lander[:, :, i + 1] = (
            v_lander[:, :, i] + (1 / 2) * (acc_old + acc_new) * time_step
        )  # Calculating velocty of rocket in timestep i+1
        acc_old = acc_new

    return time_array, r_lander, v_lander, v_terminal, Fd


time_array, r_lander, v_lander, v_terminal, Fd = landing_trajectory(
    0, (-100000 - planet_radius, 0, 0), (1000, 500, 0), 30000, 10e-4
)

theta = np.linspace(0, 2 * np.pi, 1000)
circle_x = planet_radius * np.cos(theta)
circle_y = planet_radius * np.sin(theta)
planet_surface = np.array((circle_x, circle_y, np.zeros_like(circle_x)))

plt.scatter(0, 0)
plt.scatter(r_lander[0, 0, 0], r_lander[1, 0, 0])
plt.plot(circle_x, circle_y, label="Planet surface")
plt.plot(r_lander[0, 0, :], r_lander[1, 0, :])
plt.axis("equal")
plt.xlabel("x pos (m)", fontsize=20)
plt.ylabel("y pos (m)", fontsize=20)
plt.legend(fontsize=20)
plt.title("Lander pos during landing", fontsize=20)
# plt.show()

absolute_v = np.linalg.norm(
    np.stack((v_lander[0, 0, :], v_lander[1, 0, :]), -1), axis=-1
)
plt.plot(time_array, absolute_v, label="Absolute vel lander")
plt.plot(time_array, v_terminal, label="Terminal vel")
plt.plot(time_array, np.full_like(v_terminal, 3), label="Velocity limit for landing")
plt.xlabel("Time (s)", fontsize=20)
plt.ylabel("Velocity (m/s)", fontsize=20)
plt.legend(fontsize=20)
plt.title("Velocity of lander vs estimated terminal velocity", fontsize=20)
# plt.show()

absolute_Fd = np.linalg.norm(np.stack((Fd[0, 0, :], Fd[1, 0, :]), -1), axis=-1)

plt.plot(time_array, absolute_Fd, label="Force from drag")
plt.plot(
    time_array, np.full_like(absolute_Fd, 250000), label="Break limit of parachute"
)
plt.xlabel("Time (s)", fontsize=20)
plt.ylabel("Force from drag (N)", fontsize=20)
plt.legend(fontsize=20)
plt.title("Dragforce on lander during landing", fontsize=20)
# plt.show()


# # Visualizing our atmospheric model
# x = np.linspace(-500, 500, 1000)  # meters along surface
# y = np.linspace(0, 2000000, 1000)  # meters above surface
# X, Y = np.meshgrid(x, y)
# ilen, jlen = np.shape(X)
# wind_field = np.zeros((ilen, jlen, 2))  # grid to fill wind vectors
# density_field = np.zeros((ilen, jlen))  # grid to fill with density scalars
# wind_drag_field = np.zeros((ilen, jlen, 2))  # grid to fill with dragforce vectors

# # Looping over all positions i meshgrid and calculating value
# for i in range(ilen):
#     for j in range(jlen):
#         wind_field[i, j] = (2 * np.pi * Y[i, j] / P) * np.array((1, 0))
#         density_field[i, j] = rho0 * np.exp(
#             -K * np.linalg.norm(Y[i, j] + planet_radius)
#         )  # expression for rho (simplified model using isotherm atmosphere)
#         wind_drag_field[i, j] = (
#             1
#             / 2
#             * density_field[i, j]
#             * Cd
#             * lander_area
#             * np.linalg.norm(wind_field[i, j])
#             * wind_field[i, j]
#         )  # calculating the force of air-resistance


# plt.quiver(
#     X[::10, ::10], Y[::10, ::10], wind_field[::10, ::10, 0], wind_field[::10, ::10, 1]
# )
# plt.xlabel("X (meters along surface)", fontsize=20)
# plt.ylabel("Y (meters above surface)", fontsize=20)
# plt.title("Atmospheric wind vector field", fontsize=20)
# plt.show()

# # Calculating the magnitude of the wind
# magnitude = np.sqrt(wind_field[:, :, 0] ** 2 + wind_field[:, :, 1] ** 2)

# # Plotting the streamline plot with color representing magnitude
# stream = plt.streamplot(
#     X,
#     Y,
#     wind_field[:, :, 0],
#     wind_field[:, :, 1],
#     color=magnitude,
#     cmap="viridis",
#     density=2,
# )
# cbar = plt.colorbar(stream.lines, label="Wind Speed (m/s)")
# plt.xlabel("X (meters along surface)", fontsize=20)
# plt.ylabel("Y (meters above surface)", fontsize=20)
# plt.title("Windspeed at different altitudes", fontsize=20)
# plt.show()


# # Contour plot of atmospheric density
# plt.contourf(X, Y, density_field, cmap="viridis", levels=20)
# plt.colorbar(label="Density (kg/m^3)", fontsize=20)
# plt.xlabel("X (meters along surface)", fontsize=20)
# plt.ylabel("Y (meters above surface)", fontsize=20)
# plt.title("Atmospheric density at different altitudes", fontsize=20)
# plt.show()

# plt.quiver(
#     X[::10, ::10],
#     Y[::10, ::10],
#     wind_drag_field[::10, ::10, 0],
#     wind_drag_field[::10, ::10, 1],
# )
# plt.xlabel("X (meters along surface)", fontsize=20)
# plt.ylabel("Y (meters above surface)", fontsize=20)
# plt.title("Wind drag at different altitudes", fontsize=20)
# plt.show()
