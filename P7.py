########## Ikke kodemal #############################
import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem
from ast2000tools.shortcuts import SpaceMissionShortcuts
from P1B import Engine
from p4 import spacecraft_triliteration, calculate_velocity_from_doppler, find_phi
from part3 import generalized_launch_rocket

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
parachute_area = 24.5  # m^2
lander_and_parachute_area = lander_area + parachute_area
planet_mass = system.masses[1] * SM  # kg
planet_radius = system.radii[1] * 1000  # radius in meters
homeplanet_radius = system.radii[0] * 1000  # radius homeplanet in meters
g0 = (G * planet_mass) / planet_radius**2  # gravitational acceleration at surface
mu = 22  # mean molecular weight
T0 = 296.9  # surface temperature in kelvin found in part 3
K = (2 * g0 * mu * m_H2) / (k_B * T0)
rho0 = system.atmospheric_densities[1]  # density of atmosphere at surface
P = system.rotational_periods[1] * (
    60 * 60 * 24
)  # rotational period of planet in seconds
Cd = 1  # drag coeffisient of lander
rotation_matrix = np.array([[0, -1], [1, 0]])

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

falcon_engine = Engine(
    N=2 * 10**4, L=3.775 * 10e-8, n_A=1, T=3300, t_c=10e-11, dt=10e-14
)


#################################################################
# #              Simulating landing                           # #
#################################################################


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
        0, total_simulation_time, time_step
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
        -K * np.abs(np.linalg.norm(r_lander[:, :, 0]) - planet_radius)
    )  # expression for rho (simplified model using isotherm atmosphere)

    v_terminal[0] = np.sqrt(
        G
        * (2 * lander_mass * planet_mass)
        / (rho * lander_area * (np.linalg.norm(r_lander[:, :, 0])) ** 2)
    )  # Calculating initial terminal velocity

    wind_magnitude = (
        2 * np.pi * np.abs(np.linalg.norm(r_lander[:, :, 0]) - planet_radius) / P
    )  # the wind from planet rotation at radius R from center
    wind = wind_magnitude * np.dot(
        rotation_matrix, r_lander[:, :, 0] / np.linalg.norm(r_lander[:, :, 0])
    )  # calculating the atmospheric wind

    v_drag = -v_lander[:, :, 0] + wind  # calculating the total drag-veloctiy

    Fd[:, :, 0] = (
        1 / 2 * rho * Cd * lander_area * np.square(v_drag)
    )  # calculating the force of air-resistance
    a_drag = Fd[:, :, 0] / lander_mass  # acceleration from drag
    a_planet = ((-G * planet_mass) * r_lander[:, :, 0]) / (
        np.linalg.norm(r_lander[:, :, 0]) ** 3
    )  # Sets initial acceleration from planet according to N.2 law
    acc_old = a_planet + a_drag  # total acceleration

    deploy_index = None
    # Leapfrog-loop
    for i in range(0, len(time_array) - 1):
        print(f"Calculating {i}/{len(time_array) - 1}")
        if np.linalg.norm(r_lander[:, :, i]) <= planet_radius + 2000:
            lander_area = lander_and_parachute_area
            if deploy_index == None:
                deploy_index = i

        if np.linalg.norm(r_lander[:, :, i]) <= planet_radius or np.linalg.norm(
            r_lander[:, :, i]
        ) > 2 * np.linalg.norm(r_lander[:, :, 0]):
            break_index = i
            break

        r_lander[:, :, i + 1] = (
            r_lander[:, :, i]
            + v_lander[:, :, i] * time_step
            + (acc_old * time_step**2) / 2
        )  # Rocket pos at time i+1

        rho = rho0 * np.exp(
            -K * np.abs(np.linalg.norm(r_lander[:, :, i]) - planet_radius)
        )  # expression for rho (simplified model using isotherm atmosphere)
        wind_magnitude = (
            2 * np.pi * np.abs(np.linalg.norm(r_lander[:, :, i]) - planet_radius) / P
        )  # the wind from planet rotation at radius R from center
        wind = wind_magnitude * np.dot(
            rotation_matrix, r_lander[:, :, i] / np.linalg.norm(r_lander[:, :, i])
        )  # calulating atmospheric wind
        if rho > 1:
            v_terminal[i] = np.sqrt(
                G
                * (2 * lander_mass * planet_mass)
                / (rho * lander_area * (np.linalg.norm(r_lander[:, :, i])) ** 2)
            )  # calculating terminal-velocity
        v_drag = -v_lander[:, :, i] + wind  # calculating drag velocity

        Fd[:, :, i] = (
            1 / 2 * rho * Cd * lander_area * np.square(v_drag)
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
    relative_angle = np.arcsin(r_lander[1, 0, break_index] / planet_radius)

    planet_rotation_angle = (initial_time + time_array[break_index] / P) * 2 * np.pi - (
        initial_time + time_array[break_index] // P
    ) * 2 * np.pi
    touchdown_angle = planet_rotation_angle + relative_angle
    return (
        time_array,
        r_lander,
        v_lander,
        v_terminal,
        Fd,
        deploy_index,
        break_index,
        touchdown_angle,
    )


# Initial position and velocity after using shortcut to stable orbit:
# 1 million meters above surface
# Position: (6.68704e+06, 0, 0) m
# Velocity: (0, 6250.65, 0) m/s
# 5 milion meters above surface
# Position: (1.0687e+07, 0, 0) m
# Velocity: (0, 4944.39, 0) m/s

# (
#     time_array,
#     r_lander,
#     v_lander,
#     v_terminal,
#     Fd,
#     deploy_index,
#     break_index,
#     touchdown_angle,
# ) = landing_trajectory(
#     2.53 * sec_per_year, (1.0687e07, 0, 0), (0, 4944.39 * 0.5, 0), 10000, 10e-2
# )

# theta = np.linspace(0, 2 * np.pi, 1000000)
# circle_x = planet_radius * np.cos(theta)
# circle_y = planet_radius * np.sin(theta)
# planet_surface = np.array((circle_x, circle_y, np.zeros_like(circle_x)))

# plt.scatter(0, 0)
# plt.scatter(r_lander[0, 0, 0], r_lander[1, 0, 0], label="Lander launch")
# plt.scatter(
#     r_lander[0, 0, deploy_index],
#     r_lander[1, 0, deploy_index],
#     label="Parachute deployment",
# )
# plt.scatter(
#     r_lander[0, 0, break_index],
#     r_lander[1, 0, break_index],
#     label="Touchdown!",
# )
# plt.plot(circle_x, circle_y, label="Planet surface")
# plt.plot(r_lander[0, 0, : break_index + 1], r_lander[1, 0, : break_index + 1])
# plt.axis("equal")
# plt.xlabel("x pos (m)", fontsize=20)
# plt.ylabel("y pos (m)", fontsize=20)
# plt.xticks(size=20)
# plt.yticks(size=20)
# plt.legend(fontsize=20)
# plt.title(
#     f"Lander-position, touchdown at touchdown at azimuthangle: {touchdown_angle:.2f} radians",
#     fontsize=20,
# )
# plt.show()

# absolute_v = np.linalg.norm(
#     np.stack(
#         (v_lander[0, 0, : break_index + 1], v_lander[1, 0, : break_index + 1]), -1
#     ),
#     axis=-1,
# )
# plt.plot(time_array[: break_index + 1], absolute_v, label="Absolute vel lander")
# plt.scatter(
#     time_array[deploy_index], absolute_v[deploy_index], label="Parachute deployment"
# )
# plt.plot(
#     time_array[: break_index + 1], v_terminal[: break_index + 1], label="Terminal vel"
# )
# plt.plot(
#     time_array[: break_index + 1],
#     np.full_like(time_array[: break_index + 1], 3),
#     label="Velocity limit for landing",
# )
# plt.xlabel("Time (s)", fontsize=20)
# plt.ylabel("Velocity (m/s)", fontsize=20)
# plt.xticks(size=20)
# plt.yticks(size=20)
# plt.legend(fontsize=20)
# plt.title("Velocity of lander vs estimated terminal velocity", fontsize=20)
# plt.show()

# absolute_Fd = np.linalg.norm(
#     np.stack((Fd[0, 0, : break_index + 1], Fd[1, 0, : break_index + 1]), -1), axis=-1
# )

# plt.plot(time_array[: break_index + 1], absolute_Fd, label="Force from drag")
# plt.scatter(
#     time_array[deploy_index], absolute_Fd[deploy_index], label="Parachute deployment"
# )
# plt.plot(
#     time_array[: break_index + 1],
#     np.full_like(absolute_Fd, 250000),
#     label="Break limit of parachute",
# )
# plt.xlabel("Time (s)", fontsize=20)
# plt.ylabel("Force from drag (N)", fontsize=20)
# plt.xticks(size=20)
# plt.yticks(size=20)
# plt.legend(fontsize=20)
# plt.title("Dragforce on lander during landing", fontsize=20)
# plt.show()


#################################################################
# #              Visualizing atmosphere                       # #
#################################################################
x = np.linspace(-500, 500, 1000)  # meters along surface
y = np.linspace(1, 1000000, 1000)  # meters above surface
X, Y = np.meshgrid(x, y)
ilen, jlen = np.shape(Y)
wind_field = np.zeros((ilen, jlen, 2))  # grid to fill wind vectors
density_field = np.zeros((ilen, jlen))  # grid to fill with density scalars
wind_drag_field = np.zeros((ilen, jlen, 2))  # grid to fill with dragforce vectors
print(planet_radius)
# Looping over all positions i meshgrid and calculating value
for i in range(ilen):
    for j in range(jlen):
        wind_field[i, j] = (2 * np.pi * np.abs(Y[i, j] - planet_radius) / P) * np.array(
            (-1, 0)
        )
        density_field[i, j] = rho0 * np.exp(
            -K * np.abs(np.linalg.norm(Y[i, j]) - planet_radius)
        )  # expression for rho (simplified model using isotherm atmosphere)
        wind_drag_field[i, j] = (
            1
            / 2
            * density_field[i, j]
            * Cd
            * lander_area
            * np.linalg.norm(wind_field[i, j])
            * wind_field[i, j]
        )  # calculating the force of air-resistance


plt.quiver(
    X[::10, ::10], Y[::10, ::10], wind_field[::10, ::10, 0], wind_field[::10, ::10, 1]
)
plt.xlabel("X (meters along surface)", fontsize=20)
plt.ylabel("Y (meters above surface)", fontsize=20)
plt.title("Atmospheric wind vector field", fontsize=20)
plt.show()

# Calculating the magnitude of the wind
magnitude = np.sqrt(wind_field[:, :, 0] ** 2 + wind_field[:, :, 1] ** 2)

# Plotting the streamline plot with color representing magnitude
stream = plt.streamplot(
    X,
    Y,
    wind_field[:, :, 0],
    wind_field[:, :, 1],
    color=magnitude,
    cmap="viridis",
    density=2,
)
cbar = plt.colorbar(stream.lines, label="Wind Speed (m/s)")
plt.xlabel("X (meters along surface)", fontsize=20)
plt.ylabel("Y (meters above surface)", fontsize=20)
plt.title("Windspeed at different altitudes", fontsize=20)
plt.show()


# Contour plot of atmospheric density
plt.contourf(X, Y, density_field, cmap="viridis", levels=20)
plt.colorbar(label="Density (kg/m^3)")
plt.xlabel("X (meters along surface)", fontsize=20)
plt.ylabel("Y (meters above surface)", fontsize=20)
plt.title("Atmospheric density at different altitudes", fontsize=20)
plt.show()

plt.quiver(
    X[::10, ::10],
    Y[::10, ::10],
    wind_drag_field[::10, ::10, 0],
    wind_drag_field[::10, ::10, 1],
)
plt.xlabel("X (meters along surface)", fontsize=20)
plt.ylabel("Y (meters above surface)", fontsize=20)
plt.title("Wind drag at different altitudes", fontsize=20)
plt.show()

# #################################################################
# # #             Launching using best launchtime               # #
# #################################################################
# best_launch_time_dt05 = 0.8368368368368369  # years
# (
#     altitude,
#     vertical_velocity,
#     total_time,
#     fuel_weight,
#     solar_x_pos,
#     solar_y_pos,
#     solar_x_vel,
#     solar_y_vel,
# ) = generalized_launch_rocket(
#     falcon_engine,
#     fuel_weight=165000,
#     launch_theta=np.pi / 2,
#     launch_phi=6.283185307179586,
#     launch_time=best_launch_time_dt05,
#     dt=0.01,
# )

#################################################################
# #              Updating space_mission-instance              # #
#################################################################

# time_diff = np.abs(orbit_0[0] - best_launch_time_dt05)
# least_time_diff = np.min(time_diff)
# idx_ = np.where(time_diff == least_time_diff)[0]
# initial_solar_x_pos = orbit_0[1][idx_] + (
#     (
#         (homeplanet_radius / Au)
#         * np.cos((np.pi / 2) - np.pi / 2)
#         * np.cos(6.283185307179586)
#     )
# )  # starting y-postion in Au
# initial_solar_y_pos = orbit_0[2][idx_] + (
#     (
#         (homeplanet_radius / Au)
#         * np.cos((np.pi / 2) - np.pi / 2)
#         * np.sin(6.283185307179586)
#     )
# )  # starting y-postion in Au
# mission.set_launch_parameters(
#     thrust=falcon_engine.thrust,
#     mass_loss_rate=falcon_engine.total_fuel_constant,
#     initial_fuel_mass=165000,
#     estimated_launch_duration=446.7099999963486,
#     # launch_position=[initial_solar_x_pos[0], initial_solar_y_pos[0]],
#     launch_position=[initial_solar_x_pos[0], initial_solar_y_pos[0]],
#     time_of_launch=orbit_0[0, idx_],
# )
# mission.launch_rocket()
# mission.verify_launch_result(
#     (solar_x_pos[0], solar_y_pos[0])
# )  # verifies that the calculated launch-results are correct
# distances = mission.measure_distances()
# takenimage = mission.take_picture()
# mesured_dopplershifts = mission.measure_star_doppler_shifts()
# pos_after_launch = spacecraft_triliteration(
#     best_launch_time_dt05 + total_time / sec_per_year, distances
# )
# vel_after_launch = calculate_velocity_from_doppler(
#     mesured_dopplershifts[0], mesured_dopplershifts[1]
# )
# angle_after_launch = find_phi("sky_picture.png")
# mission.verify_manual_orientation(
#     pos_after_launch, vel_after_launch, angle_after_launch
# )


#################################################################
# #                      Landing!                             # #
#################################################################
# time_of_least_distance2 = 2.53
# code_stable_orbit = 75980
# shortcut = SpaceMissionShortcuts(mission, [code_stable_orbit])
# shortcut.place_spacecraft_in_stable_orbit(
#     time_of_least_distance2, 5000000, 0, 1
# )  # <----- Using shortcut to stable orbit

# land = mission.begin_landing_sequence()
# land.orient()
# land.start_video()
# land.look_in_direction_of_planet(
#     planet_idx=1, relative_polar_angle=0, relative_azimuth_angle=0
# )
# land.fall(2500)
# land.look_in_direction_of_planet(
#     planet_idx=1, relative_polar_angle=0, relative_azimuth_angle=np.pi / 10
# )
# land.fall(2500)
# land.look_in_direction_of_planet(
#     planet_idx=1, relative_polar_angle=0, relative_azimuth_angle=np.pi - (np.pi / 10)
# )
# land.fall(2500)
# land.look_in_direction_of_planet(
#     planet_idx=1, relative_polar_angle=np.pi / 10, relative_azimuth_angle=0
# )
# land.fall(2500)


# orient = land.orient()
# vel = orient[2]

# land.adjust_parachute_area(24.5)
# # print(land.orient())
# land.look_in_direction_of_planet(
#     planet_idx=1, relative_polar_angle=0, relative_azimuth_angle=0
# )
# land.launch_lander(-0.05 * vel)
# # land.launch_lander((-500000, 0, 0))
# land.fall(2000)
# land.deploy_parachute()
# land.fall(10000)
# land.finish_video(
#     filename="Lander_video3.xml", number_of_frames=3000, radial_camera_offset=0.0
# )
