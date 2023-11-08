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
lambda_0 = 656.3  # wavelength of the Hα spectral line from restframe in nanometers
m_H2 = const.m_H2
k_B = const.k_B


def flux_sigma():  # Henter ut informasjon om fluksen og standardavviket for bølgelengdene i spektrallinjene.
    wavelength = (
        []
    )  # Lager tomme lister som skal fylles ut med info fra spektrallinjene
    flux = []
    sigma = []
    # flux_data = open(
    #     r"C:\Users\axlkl\AST2000\spectrum_seed63_600nm_3000nm.txt", "r"
    # )  # Åpner filen med info fra spektrallinjene ### Må fjerne navn fra fil
    flux_data = open(
        r"Data/spectrum_seed63_600nm_3000nm.txt", "r"
    )  # <--- legg fil i data-mappen
    for line in flux_data:  # Går gjennom hver linje i filen. 2 verdier per linje.
        line = line.strip()  # Fjerner mellomrom og div.
        line = line.split()  # Separerer de to verdiene
        wavelength.append(float(line[0]))  # Legger til verdiene i de tomme listene.
        flux.append(float(line[1]))
    flux_data.close()  # Lukker filen
    # flux_data = open(
    #     r"C:\Users\axlkl\AST2000\sigma_noise.txt", "r"
    # )  # Åpner filen med info fra spektrallinjene ### Må fjerne navn fra fil
    flux_data = open(r"Data/sigma_noise.txt", "r")  # <--- legg fil i data-mappen
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


plot_flux()


def plot_sigma():  # Plotter standardavviket til fluksen mot bølgelengdene
    wavelength, flux, sigma = flux_sigma()
    plt.plot(wavelength, sigma)  # Plotter standardavvikene.
    plt.xlabel("Nanometer", fontsize=17)
    plt.ylabel("Standardavvik", fontsize=17)
    plt.xticks(fontsize="17")
    plt.yticks(fontsize="17")
    plt.title("Standardavvik for hver bølgelengde", fontsize=17)
    plt.show()


plot_sigma()


# def chi_squared():
#     wavelength, flux, sigma = flux_sigma()


# chi_squared()
