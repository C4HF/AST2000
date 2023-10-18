########## Ikke kodemal ##################
import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem
from scipy.stats import norm
from numba import njit
from P1B import Engine
from P2 import simulate_orbits
import h5py

utils.check_for_newer_version()

np.random.seed(
    10
)  # setting random seed to ensure repeatable results when using np.random
seed = 57063  # our solarsystem seed
system = SolarSystem(seed)  # initiating SolarSystem using our seed
mission = SpaceMission(seed)  # initiating SpaceMission using our seed

"""Codeblock to fetch solar-system constants and values:"""
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

""" Mass constants"""
SM = 1.9891 * 10 ** (30)  # Solar masses in kg
m_H2 = const.m_H2
star_mass = system.star_mass  # 0.25361200295275615
homeplanet_mass = system._masses[0] * SM  # homeplanet mass in kg
dry_rocket_mass = mission.spacecraft_mass  # kg

""" Postions and velocities"""
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

""" Other constants"""
k_B = const.k_B
Au = 149597870700  # Meters
G = 6.6743 * (10 ** (-11))  # Gravitational constant
# G = 4 * (np.pi) ** 2  # Gravitational constant for Au
crosssection_rocket = mission.spacecraft_area  # m**2
homeplanet_radius = system._radii[0] * 1000  # homeplanet radius in m
star_radius = system.star_radius  # 239265.2554658649
number_of_planets = system.number_of_planets  # 7


"""Codeblock used to fetch old orbits files from h5. Not used anymore:"""
"""
# filenames = [
#     "orbit0.h5",
#     "orbit1.h5",
#     "orbit2.h5",
#     "orbit3.h5",
#     "orbit4.h5",
#     "orbit5.h5",
#     "orbit6.h5",
# ]
# for i, filename in enumerate(filenames):
#     h5f = h5py.File(filename, "r")
#     globals()[f"orbit_{i}"] = h5f["dataset_1"][:]
#     h5f.close()
"""

# Fetching data from excact-orbit-files and stores in variables in this script:
with np.load("planet_trajectories.npz") as f:
    times = f["times"]
    exact_planet_positions = f["planet_positions"]

for i, planet in enumerate(exact_planet_positions):
    globals()[f"orbit_{i}"] = (
        times,
        exact_planet_positions[0][i],
        exact_planet_positions[1][i],
    )


def generalized_launch_rocket(
    engine,
    fuel_weight,
    launch_theta,
    launch_phi,
    launch_time,
    dt=0.01,
):
    """Funksjonen tar inn instans av engine, start-fuel-vekt, ønsket hastighet,
    vinkel-posisjon mellom nordpol/sorpol (launch_theta), vinkelposisjon langs
    ekvator på planeten med vinkel null langs x-aksen (launch-phi) og oppskytningstidspunkt T i jordår fra 0-3.
    Regner deretter ut akselereasjon med hensyn på gravitasjon og regner ut hastighet og posisjon.
    Funksjonen returnerer høyde over jordoverflaten, vertikal-hastighet, total-tid, resterende drivstoffvekt
    samt xy-posisjon og xy-hastighet i forhold til stjernen i solsystemet vårt."""
    thrust = engine.thrust  # the thrust-force of the engine in N
    total_fuel_constant = (
        engine.total_fuel_constant
    )  # how many kg fuel burnt per secon kg/s
    sec_per_year = 60 * 60 * 24 * 365  # a conversion constant for years to seconds
    homeplanet_radius_Au = (
        homeplanet_radius / Au
    )  # Converting homeplanetradius in meters to Au
    rotational_velocity = np.abs(
        (2 * np.pi * (homeplanet_radius_Au) * np.cos((np.pi / 2) - launch_theta))
        / (home_planet_rotational_period / 365)
    )  # the velocity contribution from the planets rotation in Au/yr
    # Finding the closest idx in the time-array of the desired launchtime:
    time_diff = np.abs(orbit_0[0] - launch_time)
    least_time_diff = np.min(time_diff)
    idx = np.where(time_diff == least_time_diff)[0]
    solar_x_pos = orbit_0[1][idx] + (
        (
            (homeplanet_radius_Au)
            * np.cos((np.pi / 2) - launch_theta)
            * np.cos(launch_phi)
        )
    )  # starting x-postion in Au
    solar_y_pos = orbit_0[2][idx] + (
        (
            (homeplanet_radius_Au)
            * np.cos((np.pi / 2) - launch_theta)
            * np.sin(launch_phi)
        )
    )  # starting y-postion in Au
    planet_x_vel = (orbit_0[1][idx + 1] - orbit_0[1][idx]) / (
        orbit_0[0][idx + 1] - orbit_0[0][idx]
    )  # the planets starting velocty in x-direction in Au/yr
    solar_x_vel = planet_x_vel + rotational_velocity * (
        -np.sin(launch_phi)
    )  # the rockets starting velocity in the x-direction in Au/yr
    planet_y_vel = (orbit_0[2][idx + 1] - orbit_0[2][idx]) / (
        orbit_0[0][idx + 1] - orbit_0[0][idx]
    )  # the planets starting velocty in y-direction in Au/yr
    solar_y_vel = planet_y_vel + rotational_velocity * (
        np.cos(launch_phi)
    )  # # the rockets starting velocity in the y-direction in Au/yr
    # Code-block to set launch-parameters so that the current
    # launch-parameters are updated with AST2000-tools:
    mission.set_launch_parameters(
        thrust=falcon_engine.thrust,
        mass_loss_rate=falcon_engine.total_fuel_constant,
        initial_fuel_mass=165000,
        estimated_launch_duration=446.7099999963486,
        launch_position=[solar_x_pos[0], solar_y_pos[0]],
        time_of_launch=orbit_0[0][idx],
    )
    mission.launch_rocket()

    altitude = homeplanet_radius_Au  # setting starting altitude in Au, to be used in calculating force
    vertical_velocity = 0  # starting vertical velocity in Au/yr
    total_time = 0  # the total time of the launch procedure in s
    planet_x_pos = orbit_0[1][idx]  # storing the planets starting x-position in Au
    planet_y_pos = orbit_0[2][idx]  # storing the planets starting y-position in Au
    x_diff = homeplanet_radius_Au * np.cos(
        launch_phi
    )  # the distance in x-dircetion between planet and rocket
    y_diff = homeplanet_radius_Au * np.sin(
        launch_phi
    )  # the distance in y-direction between planet and rocket
    escape_velocity = np.sqrt(
        (2 * G * homeplanet_mass) / (altitude * Au)
    )  # the current escape-velocity at starting-altidtude

    # Euler-cromer loop to update postions of planet and rocket,
    # rocket velocity and acceleration. (Assumes constant planet-velocity).
    # The loop ends when the relative velocity between the rocket and planet
    # reaches is bigger that the escape-velocity required for that altidtude.
    while vertical_velocity < escape_velocity * (sec_per_year / Au):
        wet_rocket_mass = (
            dry_rocket_mass + fuel_weight
        )  # setting current wet_rocket_mass after burnt fuel
        F_gx = (
            (G * homeplanet_mass * wet_rocket_mass)
            / ((altitude * Au) ** 2)
            * (-x_diff / altitude)
        )  # The gravitational force from the planet on the rocket in the x-direction
        F_gy = (
            (G * homeplanet_mass * wet_rocket_mass)
            / ((altitude * Au) ** 2)
            * (-y_diff / altitude)
        )  # The gravitational force from the planet on the rocket in the y-direction
        thrust_x = thrust * np.cos(
            launch_phi
        )  # The thrustforce of the rocket in the x-direction
        thrust_y = thrust * np.sin(
            launch_phi
        )  # The thrustforce of the rocket in the y-direction
        sum_force_x = thrust_x + F_gx  # sum of forces in x-direction
        sum_force_y = thrust_y + F_gy  # sum of forces in y-direction
        solar_x_vel += (
            sum_force_x / wet_rocket_mass * dt * (sec_per_year / Au)
        )  # updating rocket velocity in x-direction in Au/yr
        solar_y_vel += (
            sum_force_y / wet_rocket_mass * dt * (sec_per_year / Au)
        )  # updating rocket velocity in y-direction in Au/yr
        solar_x_pos += solar_x_vel * dt / sec_per_year  # updating rocket x-pos in Au
        solar_y_pos += solar_y_vel * dt / sec_per_year  # updating rocket y-pos in Au
        planet_x_pos += planet_x_vel * dt / sec_per_year  # updating planet x-pos in Au
        planet_y_pos += planet_y_vel * dt / sec_per_year  # updating planet y-pos in Au
        vertical_velocity = np.sqrt(
            (solar_x_vel - planet_x_vel) ** 2 + (solar_y_vel - planet_y_vel) ** 2
        )  # Calculating the relative velocity between planet and rocket in Au
        x_diff = (
            solar_x_pos - planet_x_pos
        )  # difference in x-direction in Au between planet and rocket
        y_diff = (
            solar_y_pos - planet_y_pos
        )  # difference in y-direction in Au between planet and rocket

        altitude = np.sqrt(
            x_diff**2 + y_diff**2
        )  # Calculating the curren altitude from the planet-center in Au
        escape_velocity = np.sqrt(
            (2 * G * homeplanet_mass) / (altitude * Au)
        )  # Calculating the current escape-velocity in m/s
        fuel_weight -= (
            total_fuel_constant * dt
        )  # calculating the amount of burnt fuel in kg
        total_time += dt  # keeping track of time in s

        # Small codeblock to test if rocket is empty for fuel or if the rocket
        # uses an unreasonable amount of time to launh:
        if fuel_weight <= 0:
            break
        elif total_time > 1800:
            break

    mission.verify_launch_result(
        (solar_x_pos[0], solar_y_pos[0])
    )  # verifies that the calculated launch-results are correct
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


# Setting instance of our latest Engine. To be used when launching.
falcon_engine = Engine(
    N=2 * 10**4, L=3.775 * 10e-8, n_A=1, T=3300, t_c=10e-11, dt=10e-14
)
"""Codeblock to test different launch agles and launch times."""
"""
phi_arrassss = np.arange(0, np.pi / 2 + np.pi / 5, np.pi / 5)
different_times = np.arange(0, 1.4, 0.7)
for phi in phi_arrassss:
    for time in different_times:
        print("---------------")
        print(f"Launch phi (radians): {phi / np.pi}")
        print(f"Launch time (years): {time}")
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
            launch_phi=phi,
            launch_time=time,
            dt=0.01,
        )
        print(
            f"----------------------\nLaunch results:\n Total launch time (s): {total_time}\n Remaining fuel (kg): {fuel_weight} \n Solar-xy-pos (Au): ({solar_x_pos}, {solar_y_pos}) \n Solar-xy-vel (Au/yr): ({solar_x_vel}, {solar_y_vel})\n----------------------"
        )

"""
"""Codeblock to verify simulated orbits."""
"""
step = 1
mission.verify_planet_positions(
    simulation_duration=3,
    planet_positions=[
        [
            orbit_0[1, ::step],
            orbit_1[1, ::step],
            orbit_2[1, ::step],
            orbit_3[1, ::step],
            orbit_4[1, ::step],
            orbit_5[1, ::step],
            orbit_6[1, ::step],
        ],
        [
            orbit_0[2, ::step],
            orbit_1[2, ::step],
            orbit_2[2, ::step],
            orbit_3[2, ::step],
            orbit_4[2, ::step],
            orbit_5[2, ::step],
            orbit_6[2, ::step],
        ],
    ],
)
"""
"""Codeblock to generate orbit-video."""
"""
mission.generate_orbit_video(
    times=orbit_0[0],
    planet_positions=[
        [
            orbit_0[1],
            orbit_1[1],
            orbit_2[1],
            orbit_3[1],
            orbit_4[1],
            orbit_5[1],
            orbit_6[1],
        ],
        [
            orbit_0[2],
            orbit_1[2],
            orbit_2[2],
            orbit_3[2],
            orbit_4[2],
            orbit_5[2],
            orbit_6[2],
        ],
    ],
)
"""

"""Codeblock to create som plots of some rocket launches."""
"""
print(
    f"----------------------\nLaunch results2:\n Total launch time (s): {total_time2}\n Remaining fuel (kg): {fuel_weight2} \n Solar-xy-pos (Au): ({solar_x_pos2}, {solar_y_pos2}) \n Solar-xy-vel (Au/yr): ({solar_x_vel2}, {solar_y_vel2})\n----------------------"
)
# print(f"---------------\nEngine performance:\nThrust (N): {falcon_engine.thrust}")
# print("Fuel weight (kg): 165000 ")
# print("")
# print(
#     f"Initial thrust/kg (N/kg): {falcon_engine.thrust / (165000 + dry_rocket_mass):.3f}"
# )
# print(f"Total fuel constant (kg/s): {falcon_engine.total_fuel_constant}")
# print(
#     f"Thrust/total fuel constant (Ns/kg): {falcon_engine.thrust / falcon_engine.total_fuel_constant:.3f}"
# )
# print(f"Simulated pressure (pa): {falcon_engine.simulated_average_pressure:.3f}")
# print(f"Expected pressure (pa): {falcon_engine.analytical_expected_pressure:.3f}")
# print(f"Simulated total energy (J): {falcon_engine.simulated_total_energy}")
# print(f"Simulated energy (J): {falcon_engine.simulated_average_energy}")
# print(f"Analytical expected energy(J): {falcon_engine.analytical_expected_energy}")
# print(f"Density (N / (m**3) = {falcon_engine.n:.3f}\n-----------------------")
plt.plot(orbit_0[1], orbit_0[2], ls="--", label="Orbit homeplanet")
# # plt.plot((0, solar_x_pos), (0, solar_y_pos))
plt.scatter(solar_x_pos, solar_y_pos, label="Rocket")
plt.scatter(
    solar_x_pos + solar_x_vel * 10e-5,
    solar_y_pos + solar_y_vel * 10e-5,
    label="Rocket delta",
)
# plt.plot((0, solar_x_pos2), (0, solar_y_pos2))
plt.scatter(solar_x_pos2, solar_y_pos2, label="Rocket2")
plt.scatter(
    solar_x_pos2 + solar_x_vel2 * 10e-5,
    solar_y_pos2 + solar_y_vel2 * 10e-5,
    label="Rocket 2 delta",
)
plt.xlabel("Au", fontsize=20)
plt.ylabel("Au", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("Testing rocket_launch codes", fontsize=20)
plt.legend(fontsize=20)
plt.grid()
plt.show()
"""
