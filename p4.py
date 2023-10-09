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
#from P2 import simulate_orbits
#import h5py
#from part3 import generalized_launch_rocket

utils.check_for_newer_version()

np.random.seed(10)
seed = 57063
system = SolarSystem(seed)
mission = SpaceMission(seed)
Au = 149597870700  # Meters
SM = 1.9891 * 10 ** (30)  # Solar masses in kg
sec_per_year = 60 * 60 * 24 * 365
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
home_planet_initial_pos = (
    system._initial_positions[0][0],
    system._initial_positions[1][0],
)  # homeplanet initial pos in Au
homeplanet_radius = system._radii[0] * 1000  # homeplanet radius in m
# Our engine
falcon_engine = Engine(
    N=2 * 10**4, L=3.775 * 10e-8, n_A=1, T=3300, t_c=10e-11, dt=10e-14
)

from PIL import Image
def Images():   #A2
    img = Image.open(r'C:\Users\axlkl\Downloads\sample0000.png') # Open example picture
    pixels = np.array(img) # png into numpy array
    width = len(pixels[0, :])  #pixels comes in [y, x], where x is the width.
    #print(pixels, len(pixels), width) #Picture is 480(y) x 640(x) pixels.
    # redpixs = [(255, 0, 0) for i in range(width)] # Array of red pixels
    # pixels[240, :] = redpixs # Insert into line 500
    # img2 = Image.fromarray(pixels) 
    # img2.save('exampleWithRedLine.png') # Make new png with red line
    alpha = np.deg2rad(70)  #Turns degrees to radians
    phi = 0 #From task
    theta = np.pi / 2   #From task. Solar system plane
    XY_max_min = np.array([1,-1]) * ((2*np.sin(alpha / 2))/(1+np.cos(alpha / 2)))   #Stereographic projection[ 0.63059758 -0.63059758]
    
def NewPhi(png): #B
    img = Image.open(png) # Open example picture
    pixels = np.array(img) # png into numpy array
    #return newphi

# Setting launch parameters and checking launch results ##
mission.set_launch_parameters(
    thrust=falcon_engine.thrust,
    mass_loss_rate=falcon_engine.total_fuel_constant,
    initial_fuel_mass = 165000,
    estimated_launch_duration = 448.02169995917336,
    launch_position=[
        home_planet_initial_pos[0] + homeplanet_radius / Au,
        home_planet_initial_pos[1],
    ],
    time_of_launch=0,
)
mission.launch_rocket()
mission.verify_launch_result([0.06590544416834804, 0.00017508613228451168])
distances = mission.measure_distances()

# Fetching data from orbit-files:
filenames = [
    "orbit0.h5",
    "orbit1.h5",
    "orbit2.h5",
    "orbit3.h5",
    "orbit4.h5",
    "orbit5.h5",
    "orbit6.h5",
]
for i, filename in enumerate(filenames):
    h5f = h5py.File(filename, "r")
    globals()[f"orbit_{i}"] = h5f["dataset_1"][:]
    h5f.close()


def spacecraft_triliteration(T, measured_distances):
    """Function to locate position of spacecraft using mesurements to other planets and sun."""
    # finding idx of mesurement time T
    for i, t in enumerate(orbit_0[0]):
        if math.isclose(t, T, rel_tol=10e-1):
            idx = i
            break
        else:
            continue

    star_pos = np.asarray((0, 0))
    star_distance = measured_distances[-1]
    planet_2_pos = np.asarray((orbit_2[1][idx], orbit_2[2][idx]))
    planet_2_distance = measured_distances[2]
    planet_5_pos = np.asarray((orbit_5[1][idx], orbit_5[2][idx]))
    planet_5_distance = measured_distances[5]
    theta_array = np.arange(0, 2 * np.pi, 10e-7)
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
    search_2 = np.where(np.abs((abs_diff_2 - star_distance)) < 10e-7)[0]
    search_5 = np.where((np.abs(abs_diff_5 - star_distance)) < 10e-7)[0]
    n = len(search_2)
    m = len(search_5)
    matching_position = []
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
    # Kode for å visualisere trilateriring
    # print(type(found_x_pos))
    # print(type(found_y_pos))
    # plt.plot(circle_star[0], circle_star[1], label="circle star")
    # plt.plot(circle_planet2[0], circle_planet2[1], label="circle planet 2")
    # plt.plot(circle_planet5[0], circle_planet5[1], label="circle planet 5")
    # plt.scatter(
    #     0.06590544416834804, 0.00017508613228451168, label="Rocket after launch"
    # )
    # plt.scatter(found_x_pos, found_y_pos, label="Triangulated_pos")
    # plt.xlabel("Au")
    # plt.ylabel("Au")
    # plt.title("Visualisering av trilaterering")
    # plt.legend()
    # plt.show()
    return found_x_pos, found_y_pos


# spacecraft_triliteration(448.02169995917336 / sec_per_year, distances)
