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
from P1B import Engine
from P2 import simulate_orbits
import h5py
from part3 import generalized_launch_rocket
from PIL import Image


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
lambda_0 = 656.3  # wavelength of the Hα spectral line from restframe in nanometers
delta_lambda1_sun = mission.star_doppler_shifts_at_sun[0]
delta_lambda2_sun = mission.star_doppler_shifts_at_sun[1]
star_mass = system.star_mass  # 0.25361200295275615
star_radius = system.star_radius  # 239265.2554658649
number_of_planets = system.number_of_planets  # 7

sun_doppler_shift = (
    mission.star_doppler_shifts_at_sun
)  # (-0.020473606152657177, 0.01606904976188539)
star_direction_angles = (
    mission.star_direction_angles
)  # (213.2764103110655, 149.62013634196333)


v_r_sol = c * (
    np.array(sun_doppler_shift) / lambda_0
)  # nanometers / nanometers [-9352.17539636  7340.2101567 ]

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
home_planet_initial_pos = (
    system._initial_positions[0][0],
    system._initial_positions[1][0],
)  # homeplanet initial pos in Au
homeplanet_radius = system._radii[0] * 1000  # homeplanet radius in m
# Our engine
falcon_engine = Engine(
    N=2 * 10**4, L=3.775 * 10e-8, n_A=1, T=3300, t_c=10e-11, dt=10e-14
)


def generate_image(phi):  # A2
    img = Image.open("sample0000.png")
    pixels = np.array(img)
    shape = pixels.shape  # (480, 640, 3) pixels
    length = shape[0]  # 480 pixels
    width = shape[1]  # 640 pixels
    alpha_theta = 70 * (np.pi / 180)  # FOV theta
    alpha_phi = 70 * (np.pi / 180)  # FOV phi
    theta0 = np.pi / 2  # coordinate for image center
    phi0 = phi * (np.pi / 180)  # coordinate for image center
    X_max_min = np.array([1, -1]) * (
        (2 * np.sin(alpha_phi / 2)) / (1 + np.cos(alpha_phi / 2))
    )  # [ 0.63059758 -0.63059758]
    Y_max_min = np.array([1, -1]) * (
        (2 * np.sin(alpha_theta / 2)) / (1 + np.cos(alpha_theta / 2))
    )  # [ 0.63059758 -0.63059758]
    x_array = np.linspace(
        -0.63059758, 0.63059758, 640
    )  # creating x-array usin xmin/max
    y_array = np.linspace(
        -0.63059758, 0.63059758, 480
    )  # creating y-array using ymin/max
    # y_array = np.linspace(0.63059758, -0.63059758, 480)
    xv, yv = np.meshgrid(x_array, y_array)  # meshgrid with shape (2, 480, 640)
    yv = np.flip(yv)
    rho = np.sqrt(xv**2 + yv**2)  # defining constants from formula
    beta = 2 * np.arctan(rho / 2)  # defining constants from formula
    theta_array = theta0 - np.arcsin(
        np.cos(beta) * np.cos(theta0) + (yv / rho) * np.sin(beta) * np.sin(theta0)
    )  # shape = (480, 640), transformation of xv to theta-angle
    phi_array = phi0 + np.arctan(
        (xv * np.sin(beta))
        / (rho * np.sin(theta0) * np.cos(beta) - yv * np.cos(theta0) * np.sin(beta))
    )  # shape = (480, 640), transformation of yv to theta-angle
    # plt.scatter(xv, yv) # show xgrid
    # plt.show()
    # plt.scatter(theta_array, phi_array) #show angelgrids
    # plt.show()
    # coordinates = np.vstack([theta_array.ravel(), phi_array.ravel()])  # shape = (2, 307200), zips coordianets
    new_image_array = np.zeros(
        ((480, 640, 3)), dtype=np.uint8
    )  # creates empty image pixel array to be filled with rgb values
    himmelkule = np.load("himmelkule.npy")  # shape = (3145728, 5)
    ## looping over all coordinates and get indices of rgb values using the get_sky_image_pixel-method.
    ## fills up the empty image array with the rgb values
    for i in range(len(new_image_array)):
        for j in range(len(new_image_array[0])):
            idx = mission.get_sky_image_pixel(theta_array[i][j], phi_array[i][j])
            r = himmelkule[idx][2]
            g = himmelkule[idx][3]
            b = himmelkule[idx][4]
            rgb = np.array([r, g, b], dtype=np.uint8)
            new_image_array[i][j] = rgb

    new_image = Image.fromarray(
        new_image_array, "RGB"
    )  # create new image using Image.fromarray()
    # new_image.show()  # shows image
    new_image.save(f"image_phi{phi}.png")


# different_phi_arrays = np.arange(0, 360, 1)
# for phi in different_phi_arrays:
#     generate_image(phi)


def find_phi(img):
    taken_image = Image.open(img)
    taken_image_pixels = np.array(taken_image)
    different_phi_arrays = np.arange(0, 360, 1)
    best_match = 0
    least_error = 100000
    for phi in different_phi_arrays:
        compare_image = Image.open(f"Direction_images/image_phi{phi}.png")
        pixels = np.array(compare_image)
        error = np.sum(
            (taken_image_pixels.astype("float") - pixels.astype("float")) ** 2
        )
        error /= float(taken_image_pixels.shape[0] * taken_image_pixels.shape[1])
        if error < least_error:
            least_error = error
            best_match = phi
        else:
            continue
    return best_match


# test_image = Image.open(f"Direction_images/image_phi{37}.png")
# pixels_test_image = np.array(test_image)
# found = find_phi(pixels_test_image)
# print(found)


# Fetching data from orbit-files:
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

# Fetching data from orbit-files and stores in variables in this script
with np.load("planet_trajectories.npz") as f:
    times = f["times"]
    exact_planet_positions = f["planet_positions"]
print(times)
for i, planet in enumerate(exact_planet_positions[0]):
    globals()[f"orbit_{i}"] = np.array(
        (
            times,
            exact_planet_positions[0][i],
            exact_planet_positions[1][i],
        )
    )


def calculate_velocity_from_doppler(delta_lambda1, delta_lambda2):
    """Function uses mesured dopplershift delta to calculate doppler-shift.
    Returns spacecraft velocity in the xy-plane."""
    phi1 = star_direction_angles[0] * (np.pi / 180)  # angle to ref-star 1 in radians
    phi2 = star_direction_angles[1] * (np.pi / 180)  # angle to ref-star 2 in radians
    sun_doppler_shift1 = sun_doppler_shift[0]
    sun_doppler_shift2 = sun_doppler_shift[1]
    vr1 = -(c * delta_lambda1) / lambda_0
    vr2 = -(c * delta_lambda2) / lambda_0
    vr1sol = -(c * sun_doppler_shift1) / lambda_0
    vr2sol = -(c * sun_doppler_shift2) / lambda_0
    vx = (vr1 - vr1sol) * np.cos(phi1) + (vr2 - vr2sol) * np.cos(phi2)
    vy = (vr1 - vr1sol) * np.sin(phi1) + (vr2 - vr2sol) * np.sin(phi2)
    return vx, vy


# vx_sun = v_r_sol[0] * np.cos(star_direction_angles[0]) + v_r_sol[1] * np.cos(
#     star_direction_angles[1]
# )
# vy_sun = v_r_sol[0] * np.sin(star_direction_angles[0]) + v_r_sol[1] * np.sin(
#     star_direction_angles[1]
# )
# phi1 = star_direction_angles[0] * (np.pi / 180)  # angle to ref-star 1 in radians
# phi2 = star_direction_angles[1] * (np.pi / 180)  # angle to ref-star 2 in radians
# sun_doppler_shift1 = sun_doppler_shift[0]
# sun_doppler_shift2 = sun_doppler_shift[1]
# vr1sol = (c * sun_doppler_shift1) / lambda_0
# vr2sol = (c * sun_doppler_shift2) / lambda_0
# vx_sun = (vr1sol) * np.cos(phi1) + (vr2sol) * np.cos(phi2)
# vy_sun = (vr1sol) * np.sin(phi1) + (vr2sol) * np.sin(phi2)
# vx, vy = calculate_velocity_from_doppler(delta_lambda1=0, delta_lambda2=0)
# print(vx, vy)
# print(vx_sun, vy_sun)


def spacecraft_triliteration(T, measured_distances):
    """Function to locate position of spacecraft using mesured distance to other planets and sun.
    Takes in time of mesurements T, and a list of mesured_distances on the form:
    mesured_distances = [distance to planet 0, distance to planet 1, (...), distance to sun].
    Returns estimated x and y pos of rocket.
    """
    """finding idx of mesurement time T"""
    for i, t in enumerate(orbit_0[0]):
        if math.isclose(t, T, rel_tol=10e-1):
            idx = i
            break
        else:
            continue
    """Storing mesured distance to star, planet 2 and planet 5"""
    star_pos = np.asarray((0, 0))
    star_distance = measured_distances[-1]
    planet_2_pos = np.asarray((orbit_2[1][idx], orbit_2[2][idx]))
    planet_2_distance = measured_distances[2]
    planet_5_pos = np.asarray((orbit_5[1][idx], orbit_5[2][idx]))
    planet_5_distance = measured_distances[5]
    theta_array = np.arange(0, 2 * np.pi, 10e-7)
    """Parametrizing circles around star, planet 2 and planet 5 at time T, with radius = mesured distance"""
    circle_star = np.asarray(
        (
            (np.cos(theta_array) * star_distance) + star_pos[0],
            (np.sin(theta_array) * star_distance) + star_pos[1],
        )
    )
    circle_planet2 = np.asarray(
        (
            np.cos(theta_array) * planet_2_distance + planet_2_pos[0],
            np.sin(theta_array) * planet_2_distance + planet_2_pos[1],
        )
    )
    circle_planet5 = np.asarray(
        (
            np.cos(theta_array) * planet_5_distance + planet_5_pos[0],
            np.sin(theta_array) * planet_5_distance + planet_5_pos[1],
        )
    )
    """Checking when distance from parametrized circle around planet 2 and planet 5 
    is equal to the distance mesured from the star. This is the point where the three
     circles intercect. """
    diff_2_star = np.asarray(
        [
            np.abs(circle_planet2[0] - star_pos[0]),
            np.abs(circle_planet2[1] - star_pos[1]),
        ]
    )
    diff_5_star = np.asarray(
        [
            np.abs(circle_planet5[0] - star_pos[0]),
            np.abs(circle_planet5[1] - star_pos[1]),
        ]
    )

    abs_diff_2 = np.sqrt(diff_2_star[0] ** 2 + diff_2_star[1] ** 2)
    abs_diff_5 = np.sqrt(diff_5_star[0] ** 2 + diff_5_star[1] ** 2)
    search_2 = np.where(np.abs((abs_diff_2 - star_distance)) < 10e-7)[
        0
    ]  # narrowing down possible idx-points on circle 2
    search_5 = np.where((np.abs(abs_diff_5 - star_distance)) < 10e-7)[
        0
    ]  # narrowing down possible idx-points on circle 5
    n = len(search_2)
    m = len(search_5)
    matching_position = []
    """Looping over all the possible x/y-positions from search 2 and search 5 and
     check if they are around the same circle-intersection. Circles intersect at two points, but
     the possible x/y positions from the two searches are only close in the same intersection."""
    for i in range(n):
        possible_x1 = circle_planet2[0][search_2[i]]
        possible_y1 = circle_planet2[1][search_2[i]]
        for j in range(m):
            possible_x2 = circle_planet5[0][search_5[j]]
            possible_y2 = circle_planet5[1][search_5[j]]
            if math.isclose(possible_x1, possible_x2, rel_tol=10e-2) and math.isclose(
                possible_y1, possible_y2, rel_tol=10e-2
            ):
                matching_position.append(
                    ((possible_x1, possible_y1), (possible_x2, possible_y2))
                )
            else:
                continue
    possible_position_array = np.asarray(matching_position)
    least_diff = 10
    most_accurate_idx = 0
    """Looping over possible intersection positions and finding the most accurate."""
    for k in range(len(possible_position_array)):
        diff_x = possible_position_array[k, 0, 0] - possible_position_array[k, 1, 0]
        diff_y = possible_position_array[k, 0, 1] - possible_position_array[k, 1, 1]
        diff = np.sqrt(diff_x**2 + diff_y**2)
        if diff < least_diff:
            least_diff = diff
            most_accurate_idx = k
    found_x_pos = float(
        (
            possible_position_array[most_accurate_idx, 0, 0]
            + possible_position_array[most_accurate_idx, 1, 0]
        )
        / 2
    )
    found_y_pos = float(
        (
            possible_position_array[most_accurate_idx, 0, 1]
            + possible_position_array[most_accurate_idx, 1, 1]
        )
        / 2
    )
    # Code to visualize trilateration:
    # print(type(found_x_pos))
    # print(type(found_y_pos))
    # plt.plot(circle_star[0], circle_star[1], ls="-", label="circle star")
    # plt.plot(circle_planet2[0], circle_planet2[1], ls="-", label="circle planet 2")
    # plt.plot(circle_planet5[0], circle_planet5[1], ls="-", label="circle planet 5")
    # plt.plot(
    #     orbit_2[1],
    #     orbit_2[2],
    #     alpha=0.2,
    #     ls="--",
    #     dashes=(10, 20),
    #     label="Orbit planet 2",
    # )
    # plt.plot(
    #     orbit_5[1],
    #     orbit_5[2],
    #     alpha=0.2,
    #     ls="--",
    #     dashes=(10, 20),
    #     label="Orbit planet 5",
    # )
    # plt.plot(
    #     (star_pos[0], star_pos[0] + star_distance),
    #     (star_pos[1], star_pos[1]),
    #     ls="-",
    #     label="Mesured distance star",
    # )
    # plt.plot(
    #     (planet_2_pos[0], planet_2_pos[0] + planet_2_distance),
    #     (planet_2_pos[1], planet_2_pos[1]),
    #     ls="-",
    #     label="Mesured distance planet 2",
    # )
    # plt.plot(
    #     (planet_5_pos[0], planet_5_pos[0] + planet_5_distance),
    #     (planet_5_pos[1], planet_5_pos[1]),
    #     ls="-",
    #     label="Mesured distance planet 5",
    # )
    # plt.scatter(
    #     0.06590544416834804, 0.00017508613228451168, label="Rocket after launch"
    # )
    # plt.scatter(found_x_pos, found_y_pos, label="Triangulated_pos")
    # plt.scatter(star_pos[0], star_pos[1], label="Star")
    # plt.scatter(planet_2_pos[0], planet_2_pos[1], label="Planet 2")
    # plt.scatter(planet_5_pos[0], planet_5_pos[1], label="Planet 5")
    # plt.xlabel("Au")
    # plt.ylabel("Au")
    # plt.title("Visualisering av trilaterering")
    # plt.legend()
    # plt.show()
    return found_x_pos, found_y_pos


# spacecraft_triliteration(448.02169995917336 / sec_per_year, distances)


# Setting launch parameters and checking launch results ##
mission.set_launch_parameters(
    thrust=falcon_engine.thrust,
    mass_loss_rate=falcon_engine.total_fuel_constant,
    initial_fuel_mass=165000,
    estimated_launch_duration=448.02169995917336,
    launch_position=[
        home_planet_initial_pos[0] + homeplanet_radius / Au,
        home_planet_initial_pos[1],
    ],
    time_of_launch=0,
)
mission.launch_rocket()
mission.verify_launch_result([0.06590544416834804, 0.00017508613228451168])
distances = mission.measure_distances()
takenimage = mission.take_picture()
mesured_dopplershifts = mission.measure_star_doppler_shifts()

# print(mesured_dopplershifts)


# Analyzing using onboard equipment
pos_after_launch = spacecraft_triliteration(448.02169995917336, distances)
vel_after_launch = calculate_velocity_from_doppler(
    mesured_dopplershifts[0], mesured_dopplershifts[1]
)
# vel_after_launch = calculate_velocity_from_doppler(
#     sun_doppler_shift[0], sun_doppler_shift[1]
# )
# ----------------------
# Launch results:
#  Total launch time (s): 448.02299999631754
#  Remaining fuel (kg): 6264.647279576635
#  Solar-xy-pos (Au): (0.06590544416834804, 0.00017508613228451168)
#  Solar-xy-vel (Au/yr): (2.413600055971701, 12.324180383036367)
# ----------------------
# print(pos_after_launch)
# print(f"Got:  {vel_after_launch}")
angle_after_launch = find_phi("sky_picture.png")

mission.verify_manual_orientation(
    pos_after_launch, vel_after_launch, angle_after_launch
)
