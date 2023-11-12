########## Ikke kodemal #############################
import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem
import pandas as pd
from scipy.optimize import minimize

# from scipy import constants


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
    flux_data = open(r"Data/spectrum_seed63_600nm_3000nm.txt", "r")
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
CO2_spectral_lines = [1400, 1660]
CH4_spectral_lines = [1660, 2200]
CO_spectral_lines = [2340]
N20_spectral_lines = [2870]

spectral_lines = [
    632,
    690,
    760,
    720,
    820,
    940,
    1400,
    1600,
    1660,
    2200,
    2340,
    2870,
]  # list of all spectral lines

vel_max = 10  # km/s
vel_min = -10  # km/s


def plot_slice_of_flux_around_spectralline(lambda_0):
    """Function to plot a slice of the observed flux in a range around expected spectrallines."""
    wavelength, flux, sigma = flux_sigma()  # getting data
    wavelength_arr = np.array(wavelength)
    flux_arr = np.array(flux)
    sigma_arr = np.array(sigma)

    # Max wavelength
    lambda_max = (
        lambda_0 * vel_max / const.c_km_pr_s
    ) + lambda_0  # Applying dopplershift
    lambda_diff_max = np.abs(wavelength_arr - lambda_max)
    least_diff_max = np.min(lambda_diff_max)
    idx_max = np.where(lambda_diff_max == least_diff_max)[0]

    # Min wavelength
    lambda_min = (
        lambda_0 * vel_min / const.c_km_pr_s
    ) + lambda_0  # Applying dopplershift
    lambda_diff_min = np.abs(wavelength_arr - lambda_min)
    least_diff_min = np.min(lambda_diff_min)
    idx_min = np.where(lambda_diff_min == least_diff_min)[0]

    # Slicing observed data to only plot between min/max values
    wavelength_slice = np.array(wavelength_arr[idx_min[0] : idx_max[0] + 1])
    flux_slice = np.array(flux_arr[idx_min[0] : idx_max[0] + 1])
    noise_slice = np.array(sigma_arr[idx_min[0] : idx_max[0] + 1]) + 1

    # Smoothing data with rolling-average for readablity
    window_size = 10  # number of points to calculate rolling average
    flux_slice_pd = pd.Series(flux_slice)
    smoothed_data = flux_slice_pd.rolling(window_size).mean()

    # Plotting data
    plt.plot(
        wavelength_slice, flux_slice, label="Observed flux", alpha=0.5, color="gray"
    )
    plt.plot(wavelength_slice, smoothed_data, label="Smoothed_data", color="red")
    plt.plot(
        wavelength_slice,
        np.full_like(wavelength_slice, 1),
        label="Base-line",
        color="black",
    )
    plt.plot(
        wavelength_slice,
        noise_slice,
        label="Noise-level + 1",
        alpha=0.6,
        linestyle="--",
    )
    plt.xlabel("Wavelentgh (NM)", fontsize=20)
    plt.ylabel("Standardized flux (LM) ", fontsize=20)
    plt.ticklabel_format(useOffset=False)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.grid(which="both", alpha=0.3)
    plt.minorticks_on()
    plt.title(f"Observed flux around: {lambda_0} NM", fontsize=20)
    plt.legend(fontsize=20, loc="upper right")
    plt.show()
    # plt.savefig(f"Data/flux_{lambda_0}_nm")


# plot_slice_of_flux_around_spectralline(lambda_0=1660)
# for spectral_line in spectral_lines:
#     plot_slice_of_flux_around_spectralline(lambda_0=spectral_line)


######################################################################
##                 This part is not working yet                     ##
######################################################################


def F_gauss_line_profile(Fmin, lambda_0, std, lambda_array):
    """The gaussian line profiel. Takes in Fmin, lambda_ lambda_0 and std."""
    return 1 + (Fmin - 1) * np.exp(-0.5 * ((lambda_array - lambda_0) / std) ** 2)


def chi_squared_minimization(params, flux_slice, sigma_slice):
    """Function to loop over range of possible values for sepctralline, create a model
    and compare it to observed data using chi-squared. The smaller chi-squared is,
    the better the model."""
    m, lambda_0, Fmin = params
    lambda_array = (
        lambda_0 * velocity / const.c_km_pr_s
    ) + lambda_0  # Expected lambda-range based on dopplershift from radial velocity
    std_array = (
        lambda_0 * np.sqrt(const.k_B * temperature) / (const.c * m)
    )  # Ecpectd std-rannge basen on expected temperature-range

    least_chi = 100000
    best_std = 0
    best_lambda = 0
    best_model = None

    ## Double for loop to test all parameters in the expected range and find the best
    ## parameters for the gaussian_line_profile.
    for lambda_i in lambda_array:
        for std_j in std_array:
            model = F_gauss_line_profile(
                Fmin, lambda_i, std_j, lambda_array
            )  # creating gaussian-model with parameters
            chi_squared = np.sum(
                ((flux_slice - model) / sigma_slice) ** 2
            )  # calculating chi_squared
            if chi_squared < least_chi:  # keeping the best results
                best_std = std_j
                best_lambda = lambda_i
                best_model = model

    return best_std, best_lambda, best_model


def get_slice(lambda_0):
    """Function returns slices of the observed data in the wavelength range where
    we ecxpect to find spectrallines. The range is calculated from the potential
    dopplershift because of the rockets radial-velocity to the planet."""
    wavelength, flux, sigma = flux_sigma()  # getting data
    wavelength_arr = np.array(wavelength)
    flux_arr = np.array(flux)
    sigma_arr = np.array(sigma)
    # Max wavelength
    lambda_max = (
        lambda_0 * vel_max / const.c_km_pr_s
    ) + lambda_0  # Calculating dopplershift
    lambda_diff_max = np.abs(wavelength_arr - lambda_max)
    least_diff_max = np.min(lambda_diff_max)
    idx_max = np.where(lambda_diff_max == least_diff_max)[0]

    # Min wavelength
    lambda_min = (
        lambda_0 * vel_min / const.c_km_pr_s
    ) + lambda_0  # Calculating dopplershift
    lambda_diff_min = np.abs(wavelength_arr - lambda_min)
    least_diff_min = np.min(lambda_diff_min)
    idx_min = np.where(lambda_diff_min == least_diff_min)[0]

    # Slicing observed data to only plot between min/max values
    wavelength_slice = np.array(wavelength_arr[idx_min[0] : idx_max[0] + 1])
    flux_slice = np.array(flux_arr[idx_min[0] : idx_max[0] + 1])
    noise_slice = np.array(sigma_arr[idx_min[0] : idx_max[0] + 1]) + 1
    return wavelength_slice, flux_slice, noise_slice


# Parameters
lambda_0 = 1660
wavelength_slice, flux_slice, noise_slice = get_slice(lambda_0)
N = len(flux_slice)
temperature = np.linspace(150, 450, N)  # Expected temperatur-range in atmosphere
velocity = np.linspace(-10, 10, N)
m = 32 * 10e-3
Fmin = 0.7
params = (m, lambda_0, Fmin)
best_std, best_lambda, best_model = chi_squared_minimization(
    params, flux_slice, noise_slice
)

## Plotting results ##
print(f"Best parameters: Std={best_std}, Lambda={best_lambda}")
plt.figure(figsize=(10, 5))

# Plot observed data
plt.subplot(1, 2, 1)
plt.plot(wavelength_slice, flux_slice, label="Observed Data")
plt.title("Observed Data")
plt.legend()

# Plot fitted model
plt.subplot(1, 2, 2)
plt.plot(wavelength_slice, best_model, label="Fitted Model", color="orange")
plt.title("Fitted Model")
plt.legend()

plt.show()
