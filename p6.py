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


def flux_sigma():  # Henter ut informasjon om fluksen og standardavviket for bølgelengdene i spektrallinjene.
    wavelength = (
        []
    )  # Lager tomme lister som skal fylles ut med info fra spektrallinjene
    flux = []
    sigma = []
    flux_data = open(
        r"Data/spectrum_seed63_600nm_3000nm.txt", "r"
    ) 
    for line in flux_data:  # Går gjennom hver linje i filen. 2 verdier per linje.
        line = line.strip()  # Fjerner mellomrom og div.
        line = line.split()  # Separerer de to verdiene
        wavelength.append(float(line[0]))  # Legger til verdiene i de tomme listene.
        flux.append(float(line[1]))
    flux_data.close()  # Lukker filen
    flux_data = open(r"Data/sigma_noise.txt", "r")  # 
    for line in flux_data:  # Går gjennom hver linje i filen. 2 verdier per linje.
        line = line.strip()  # Fjerner mellomrom og div.
        line = line.split()  # Separerer de to verdiene
        sigma.append(float(line[1]))  # Legger til verdiene i de tomme listene.
    flux_data.close()  # Lukker filen
    return (
        wavelength,
        flux,
        sigma,
    )  # Returnerer bølgelengder med tilhørende fluks og  densstandarddavik.


def plot_flux():  # Plotter fluksen mot bølgelengdene
    wavelength, flux, sigma = flux_sigma()

    plt.plot(wavelength, flux)  # Plotter spektrallinjene.
    plt.xlabel("Nanometer", fontsize=17)
    plt.ylabel("Fluks", fontsize=17)
    plt.xticks(fontsize="17")
    plt.yticks(fontsize="17")
    plt.title("Spektrallinjer", fontsize=17)
    plt.show()


# plot_flux()
# plot_flux()


def plot_sigma():  # Plotter standardavviket til fluksen mot bølgelengdene
    wavelength, flux, sigma = flux_sigma()
    plt.plot(wavelength, sigma)  # Plotter standardavvikene.
    plt.xlabel("Nanometer", fontsize=17)
    plt.ylabel("Standardavvik", fontsize=17)
    plt.xticks(fontsize="17")
    plt.yticks(fontsize="17")
    plt.title("Standardavvik for hverw bølgelengde", fontsize=17)
    plt.show()


# plot_sigma()

# Spectral lines for each gas without doppler shift in nanometers
O2_spectral_lines = [632, 690, 760]
H2O_spectral_lines = [720, 820, 940]
CO2_spectral_lines = [1400, 1600]
CH4_spectral_lines = [1660, 2200]
CO_spectral_lines = [2340]
N20_spectral_lines = [2870]

N = 1000000
temperature = np.linspace(150, 450, N)  # Expected temperatur-range in atmosphere
velocity = np.linspace(-10, 10, N)

# F = lambda Fmin, lambda_, lambda_0, std: 1 + (Fmin - 1) * np.exp(
#     -0.5 * ((lambda_ - lambda_0) / (std)) ** 2
# )


def plot_slice_of_flux_around_spectralline(m, lambda_0):
    """Function to plot a slice of the observed flux in a range around expected spectrallines."""
    lambda_array = (lambda_0 * velocity / const.c_km_pr_s) + lambda_0
    std_array = lambda_0 * np.sqrt(const.k_B * temperature) / (const.c * m)
    Fmin = 0.7
    wavelength, flux, sigma = flux_sigma()

    lambda_max = np.max(lambda_array)

    lambda_diff_max = np.abs(wavelength - lambda_max)
    least_diff_max = np.min(lambda_diff_max)
    idx_max = np.where(lambda_diff_max == least_diff_max)[0]
    lambda_min = np.min(lambda_array)
    lambda_diff_min = np.abs(wavelength - lambda_min)
    least_diff_min = np.min(lambda_diff_min)
    idx_min = np.where(lambda_diff_min == least_diff_min)[0]

    wavelength_slice = np.array(wavelength[idx_min[0] : idx_max[0] + 1])
    flux_slice = np.array(flux[idx_min[0] : idx_max[0] + 1])
    noise_slice = np.array(sigma[idx_min[0] : idx_max[0] + 1])

    plt.plot(
        wavelength_slice,
        flux_slice,
        label="observed flux",
    )

    weighet_flux1 = np.where(flux_slice > 1, flux_slice / (1 + noise_slice), flux_slice)
    weighet_flux2 = np.where(
        flux_slice < 1, flux_slice / (1 - noise_slice), weighet_flux1
    )
    plt.plot(
        wavelength_slice,
        weighet_flux2,
        label="observed flux down-weighted with noise",
    )

    plt.plot(
        wavelength_slice,
        np.full_like(wavelength_slice, 1),
        label="Normalized average flux",
    )
    plt.title(f"Plot of flux around expected spectral line: {spectral_line} NM")
    plt.legend()
    plt.show()


for spectral_line in O2_spectral_lines:
    create_gauss_model(m=1, lambda_0=spectral_line)
# plt.plot(lambda_array, F)
# print(np.shape(F))
# plt.hist(F, bins=10000)

# gauss = np.random.normal(loc=lambda_array[N // 2], scale=std_array[N // 2], size=N)
# sigma = np.linspace(-4 * std_array[N // 2], 4 * std_array[N // 2], N)
# plt.plot(sigma, gauss)
# plt.hist(gauss, bins=1000)
# plt.show()
# for lambda_ in lambda_array:
#     for std in std_array:
#         gauss = np.random.normal(loc=lambda_, scale=std, size=N)


# create_gauss_model(2.65e-26, 632)
# create_gauss_model(632, 2)
