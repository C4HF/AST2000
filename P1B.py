########## Egen kode ##################################
import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem
import math
from scipy.stats import norm

# seed = utils.get_seed("username")
seed = 57063
system = SolarSystem(seed)
mission = SpaceMission(seed)
utils.check_for_newer_version()
# @jit(nopython = True) #Optimalisering(?)

"""Parametre"""
L = 10e-7  # Bredde på boksen i meter
T = 3000  # Gassens temperatur i kelvin
N = 10**4  # Antall partikler
t_c = 10e-9  # Tid
dt = 10e-12  # Tids intervall i s'
t = np.arange(0, t_c, dt)  # Tidsarray definert vha Tid og tidssteg
m_H2 = const.m_H2
k_B = const.k_B
MB = np.sqrt(const.k_B * T / const.m_H2)

SM = 5.02739933 * 10 ** (31)  # Solar masses in kg
G = 6.6743 * 10 ** (-11)  # Gravitational constant
dry_rocket_mass = mission.spacecraft_mass
crosssection_rocket = mission.spacecraft_area
homeplanet_radius = system._radii[0] * 1000  # homeplanet radius in m
homeplanet_mass = system._masses[0] * SM  # homeplanet mass in kg
# print(system.__dir__())  ### get a list of attribute-commands


# print(homeplanet_radius)
# print(homeplanet_mass)
# for planet_idx in range(system.number_of_planets):
#     print(
#         f"Planet index{planet_idx}, type {system.types[planet_idx]} has mass {system._masses[planet_idx]} and radius {system._radii[planet_idx]}"
#     )

### Må confirme escape velocity ###############
# escape_velocity = 11882
# print(escape_velocity)
# print(np.sqrt((2 * G * homeplanet_mass) / homeplanet_radius))

"""Kode for 1B og 1C."""


class Engine:
    """Create instance of engine with N particles, L dimension small engine,
    n_A nozzle area, T temperature in Kelvin, t_c simulation time, dt simulation timestep.
    """

    def __init__(self, N: int, L: float, n_A: float, T: float, t_c: float, dt: float):
        if n_A > L**2:
            raise ValueError(
                f"n_A should be less than {L**2:.2f}. Got n_A = {n_A:.2f}."
            )
        self.N = N
        self.L = L
        self.n_A = n_A
        self.T = T
        self.t_c = t_c
        self.dt = dt
        self._simulate_small_engine()  # function is called by constructer

    def _simulate_small_engine(self):
        """Simulate engine performance."""
        N = self.N
        L = self.L
        n_A = self.n_A
        T = self.T
        t_c = self.t_c
        dt = self.dt
        l = np.sqrt(n_A)
        d = (L - l) / 2
        m_H2 = const.m_H2
        k_B = const.k_B
        MB = np.sqrt(const.k_B * T / const.m_H2)

        a = []  # Skal telle farten til partiklene som slipper ut
        rows = 3  # For vectors
        cols = N
        pos = L * np.random.rand(rows, cols)  # Particle positions
        mean = 0
        std = MB
        vel = np.random.normal(
            loc=mean, scale=std, size=(rows, cols)
        )  # loc = mean, scale = standard deviation(std)
        for i in range(
            len(vel)
        ):  # Sørger for at ingen hastigheter er negative(Sjelden feil)
            vel_0 = np.where(vel[i] == 0)[0]
            vel[i][vel_0] = vel[i - 1][vel_0]

        pressure_list = []

        for m in range(len(t)):  # tidssteg
            pos += dt * vel  # Euler cromer

            x1 = np.where(pos[0] >= L)[0]  # Ser etter kollisjoner for x
            x2 = np.where(pos[0] <= 0)[0]
            y1 = np.where(pos[1] >= L)[0]  # Ser etter kollisjoner for y
            y2 = np.where(pos[1] <= 0)[0]
            z1 = np.where(pos[2] >= L)[0]  # Ser etter kollisjoner for z
            z2 = np.where(pos[2] <= 0)[0]

            for m in range(len(z2)):
                if (
                    d < pos[0][z2[m]] < (L - d)
                ):  # Sjekker om kollisjonene for z2(xy-planet) egentlig er i utgangshullet
                    if d < pos[1][z2[m]] < (L - d):
                        a.append(
                            vel[2][z2[m]]
                        )  # Lagrer farten til partiklene som forsvinner ut. Kan brukes til beregninger
                        for i in range(
                            2
                        ):  # Flytter partikkelen til en uniformt fordelt posisjon på toppen av boksen, med samme vel.
                            pos[i][m] = L * np.random.rand()
                        pos[2][m] = L

            vel[0][x1] = -vel[0][x1]
            vel[0][x2] = -vel[0][
                x2
            ]  # Elastisk støt ved å snu farten til det motsatte i en gitt retning
            vel[1][y1] = -vel[1][y1]
            vel[1][y2] = -vel[1][y2]
            vel[2][z1] = -vel[2][z1]
            vel[2][z2] = -vel[2][z2]

            # Trykk per tidsteg
            momentum = vel * m_H2
            df = np.sum(
                (2 * np.abs(momentum[0][x1])) / dt
            )  # regner ut kraft som virker på veggen per tidssteg
            dp = df / (L * L)
            pressure_list.append(dp)

        # Trykk
        self.simulated_average_pressure = sum(pressure_list) / len(pressure_list)
        n = N / (L * L * L)
        self.analytical_expected_pressure = n * k_B * T

        # Energi
        self.simulated_kinetic_energy = 1 / 2 * m_H2 * vel**2
        self.simulated_total_energy = np.sum(self.simulated_kinetic_energy)
        self.simulated_average_energy = self.simulated_total_energy / N
        self.analytical_expected_energy = (3 / 2) * k_B * T

        # Fuel consumption
        self.tot_fuel = m_H2 * len(a)
        self.fuel_cons = (
            self.tot_fuel / t_c
        )  # fuel constant per second per small engine

        # Fremdrift
        self.P = sum(a) * m_H2  # P = mv, bruker bare v_z da de andre blir 0 totalt.
        self.F = -self.P / t_c  # F = mv / dt

        ## Utregning av total thrust og total fuel-constant
        self.number_of_engines = crosssection_rocket / (L**2)
        self.thrust = self.number_of_engines * self.F
        self.total_fuel_constant = self.fuel_cons * self.number_of_engines

        return vel, average_pressure, average_energy, F, fuel_cons


def simulate_small_engine(
    npb,
):
    """Funksjonen simulerer bevegelsen til partiklene i en liten boks.
    Tar inn variabel npb (antall partikler per boks) og returnerer
    tuppel: (vel, average_pressure, average_energy, F, fuel_cons).
    """
    a = []  # Skal telle farten til partiklene som slipper ut
    rows = 3  # For vectors
    cols = npb
    pos = L * np.random.rand(rows, cols)  # Particle positions
    mean = 0
    std = MB
    vel = np.random.normal(
        loc=mean, scale=std, size=(rows, cols)
    )  # loc = mean, scale = standard deviation(std)
    for i in range(
        len(vel)
    ):  # Sørger for at ingen hastigheter er negative(Sjelden feil)
        vel_0 = np.where(vel[i] == 0)[0]
        vel[i][vel_0] = vel[i - 1][vel_0]

    pressure_list = []

    for m in range(len(t)):  # tidssteg
        pos += dt * vel  # Euler cromer

        x1 = np.where(pos[0] >= L)[0]  # Ser etter kollisjoner for x
        x2 = np.where(pos[0] <= 0)[0]
        y1 = np.where(pos[1] >= L)[0]  # Ser etter kollisjoner for y
        y2 = np.where(pos[1] <= 0)[0]
        z1 = np.where(pos[2] >= L)[0]  # Ser etter kollisjoner for z
        z2 = np.where(pos[2] <= 0)[0]

        for m in range(len(z2)):
            if (
                L / 8 < pos[0][z2[m]] < (7 / 8) * L
            ):  # Sjekker om kollisjonene for z2(xy-planet) egentlig er i utgangshullet
                if L / 8 < pos[1][z2[m]] < (7 / 8) * L:
                    a.append(
                        vel[2][z2[m]]
                    )  # Lagrer farten til partiklene som forsvinner ut. Kan brukes til beregninger
                    for i in range(
                        2
                    ):  # Flytter partikkelen til en uniformt fordelt posisjon på toppen av boksen, med samme vel.
                        pos[i][m] = L * np.random.rand()
                    pos[2][m] = L

        vel[0][x1] = -vel[0][x1]
        vel[0][x2] = -vel[0][
            x2
        ]  # Elastisk støt ved å snu farten til det motsatte i en gitt retning
        vel[1][y1] = -vel[1][y1]
        vel[1][y2] = -vel[1][y2]
        vel[2][z1] = -vel[2][z1]
        vel[2][z2] = -vel[2][z2]

        # ax.scatter(pos[0], pos[1], pos[2])  # Plotter rakettmotoren

        # Trykk per tidsteg
        momentum = vel * m_H2
        df = np.sum(
            (2 * np.abs(momentum[0][x1])) / dt
        )  # regner ut kraft som virker på veggen per tidssteg
        dp = df / (L * L)
        pressure_list.append(dp)

    # Trykk
    average_pressure = sum(pressure_list) / len(pressure_list)
    n = N / (L * L * L)
    analytical_pressure = n * k_B * T

    # Energi
    numerical_kinetic_energy = 1 / 2 * m_H2 * vel**2
    numerical_total_energy = np.sum(numerical_kinetic_energy)
    average_energy = numerical_total_energy / N
    analytical_average_energy = (3 / 2) * k_B * T

    # Fuel consumption
    tot_fuel = m_H2 * len(a)
    fuel_cons = tot_fuel / t_c

    # Fremdrift
    P = sum(a) * m_H2  # P = mv, bruker bare v_z da de andre blir 0 totalt.
    F = -P / t_c  # F = mv / dt

    return vel, average_pressure, average_energy, F, fuel_cons


def plot_velocity_distribution(npb, bins=30):
    """Funksjonen kaller på simulate_small_engine funksjonen og danner et
    subplot som sammenligner den simulerte farten med Maxwell-Boltzmann fordelingen.
    Tar inn npb variabel (antall partikler per boks), bins (antall stolper i histogrammet).
    Returnerer ingenting"""
    (vel, average_pressure, average_energy, F, fuel_cons) = simulate_small_engine(npb)
    mean = 0
    std = MB
    x_axis = np.linspace(-4 * std, 4 * std, N)
    bins = bins
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

    ax1.hist(
        vel[0], bins=bins, alpha=0.4, density=True, label="x-velocity", color="cyan"
    )
    ax1.plot(
        x_axis,
        norm.pdf(x_axis, loc=mean, scale=std),
        color="red",
        label="MB-dist",
    )

    ax2.hist(
        vel[1], bins=bins, alpha=0.4, density=True, label="y-velocity", color="olive"
    )
    ax2.plot(
        x_axis,
        norm.pdf(x_axis, loc=mean, scale=std),
        color="red",
        label="MB-dist",
    )

    ax3.hist(
        vel[2], bins=bins, alpha=0.4, density=True, label="z-velocity", color="pink"
    )
    ax3.plot(
        x_axis,
        norm.pdf(x_axis, loc=mean, scale=std),
        color="red",
        label="MB-dist",
    )
    ax1.set_ylabel("%")
    ax1.set_xlabel("m/s")
    ax1.set_title("Velocity x-direction")
    ax1.legend(loc="upper left")
    ax2.set_xlabel("m/s")
    ax2.set_title("Velocity y-direction")
    ax2.legend(loc="upper left")
    ax3.set_xlabel("m/s")
    ax3.set_title("Velocity z-direction")
    ax3.legend(loc="upper left")
    fig.suptitle("Simulated velocity of our particles compared to Maxwell-Boltzmann")
    plt.show()


def plot_small_engine(npb=100, time=100):
    from mpl_toolkits import mplot3d  # Plotting

    fig = plt.figure()
    ax = plt.axes(projection="3d")  # For 3d plotting av rakettmotoren
    ax.plot3D([0, L], [0, 0], [0, 0], "green")  # Lager en yttre firkant på xy-planet
    ax.plot3D([L, L], [0, L], [0, 0], "green")
    ax.plot3D([0, 0], [0, L], [0, 0], "green")
    ax.plot3D([0, L], [L, L], [0, 0], "green")

    ax.plot3D(
        [0.25 * L, 0.75 * L], [0.25 * L, 0.25 * L], [0, 0], "green"
    )  # Lager en indre firkant på xy-planet (utgangshull)
    ax.plot3D([0.25 * L, 0.75 * L], [0.75 * L, 0.75 * L], [0, 0], "green")
    ax.plot3D([0.25 * L, 0.25 * L], [0.25 * L, 0.75 * L], [0, 0], "green")
    ax.plot3D([0.75 * L, 0.75 * L], [0.25 * L, 0.75 * L], [0, 0], "green")

    nr = []  # Bare til plotting underveis
    rows = 3  # For vectors
    cols = npb
    pos = L * np.random.rand(rows, cols)  # Particle positions
    mean = 0
    std = MB
    vel = np.random.normal(
        loc=mean, scale=std, size=(rows, cols)
    )  # loc = mean, scale = standard deviation(std)
    for i in range(
        len(vel)
    ):  # Sørger for at ingen hastigheter er negative(Sjelden feil)
        vel_0 = np.where(vel[i] == 0)[0]
        vel[i][vel_0] = vel[i - 1][vel_0]

    for m in range(time):  # tidssteg
        pos += dt * vel  # Euler cromer

        x1 = np.where(pos[0] >= L)[0]  # Ser etter kollisjoner for x
        x2 = np.where(pos[0] <= 0)[0]
        y1 = np.where(pos[1] >= L)[0]  # Ser etter kollisjoner for y
        y2 = np.where(pos[1] <= 0)[0]
        z1 = np.where(pos[2] >= L)[0]  # Ser etter kollisjoner for z
        z2 = np.where(pos[2] <= 0)[0]

        for m in range(len(z2)):  # Sjekker om kollisjonene for z2(xy-planet)
            if L / 4 < pos[0][z2[m]] < (3 / 4) * L:  # egentlig er i
                if L / 4 < pos[1][z2[m]] < (3 / 4) * L:  # utgangshullet
                    for i in range(2):  # Flytter partikkelen til en uniformt
                        pos[i][m] = L * np.random.rand()  # fordelt posisjon på
                    pos[2][m] = L  # toppen av boksen, med samme vel.
                    if z2[m] not in nr:  # Brukes til plotting
                        nr.append(z2[m])
        # z2 = list(z2)
        # x1 = list(x1)
        # x2 = list(x2)
        # y1 = list(y1)
        # y2 = list(y2)
        # for i in range(len(nr)):  # For plotting
        #     if nr[i] in z2:       # Av at partiklene fyker ut av boksen
        #         z2.remove(nr[i])
        #     if nr[i] in x1:
        #         x1.remove(nr[i])
        #     if nr[i] in x2:
        #         x2.remove(nr[i])
        #     if nr[i] in y1:
        #         y1.remove(nr[i])
        #     if nr[i] in y2:
        #         y2.remove(nr[i])

        vel[0][x1] = -vel[0][x1]
        vel[0][x2] = -vel[0][x2]  # Elastisk støt ved å snu
        vel[1][y1] = -vel[1][y1]  # farten til det motsatte i en gitt retning
        vel[1][y2] = -vel[1][y2]
        vel[2][z1] = -vel[2][z1]
        vel[2][z2] = -vel[2][z2]

        ax.scatter(pos[0], pos[1], pos[2])  # Plotter rakettmotoren

    plt.show()


"""Kode 1D"""


def calculate_needed_fuel(thrust, fuel_consumption, initial_rocket_mass, speed_boost):
    """Funksjonen regner ut hvor mye drivstoff vi trenger for å øke hastigheten med en ønsket mengde.
    Tar inn variabler (thrust, fuel_consumption, initial_rocket_mass, speed_boost) og returnerer
    fuel_consumed."""
    dt = 10
    start_speed = 0
    total_time = 0
    while start_speed < speed_boost:
        start_speed += (thrust / initial_rocket_mass) * dt
        initial_rocket_mass -= fuel_consumption * dt
        total_time += dt
    fuel_consumed = total_time * fuel_consumption
    return fuel_consumed


## Utregning av total thrust, total fuel-constant og total rocket mass
(vel, average_pressure, average_energy, F, fuel_cons) = simulate_small_engine(N)
estimated_fuel_weight = 4500
number_of_engines = crosssection_rocket / (L**2)
rocket_total_thrust = number_of_engines * F
total_fuel_constant = fuel_cons * number_of_engines
wet_rocket_mass = dry_rocket_mass + estimated_fuel_weight

# print("Number of engines:")
# print(number_of_engines)
# print("Total thrust per second in Newton:")
# print(rocket_total_thrust)
# print("Fuel consumed per small engine per second:")
# print(fuel_cons)
# print("Total fuel used per second of entire rocket:")
# print(total_fuel_constant)
# print("Total weight rocket")
# print(wet_rocket_mass)

# needed_fuel = calculate_needed_fuel(
#     rocket_total_thrust, total_fuel_constant, wet_rocket_mass, escape_velocity
# )
# print("Total burned fuel to perform velocity-boost")
# print(needed_fuel)

"""Kode 1E"""


def launch_rocket(fuel_weight, target_vertical_velocity, dt=10):
    """Funksjonen simulerer launch av raketten."""
    vel, average_pressure, average_energy, F, fuel_cons = simulate_small_engine(N)
    altitude = 0
    vertical_velocity = 0
    total_time = 0

    wet_rocket_mass = dry_rocket_mass + fuel_weight
    F_g = (G * homeplanet_mass * wet_rocket_mass) / (
        homeplanet_radius + altitude
    )  # The gravitational force
    rocket_thrust_gravitation_diff = (number_of_engines * F) - F_g
    total_fuel_constant = fuel_cons * number_of_engines

    while vertical_velocity < target_vertical_velocity:
        vertical_velocity += (rocket_thrust_gravitation_diff / wet_rocket_mass) * dt
        altitude += vertical_velocity * dt
        fuel_weight -= total_fuel_constant * dt
        total_time += dt

        if fuel_weight <= 0:
            break
        elif total_time > 1800:
            break
        elif altitude < 0:
            break
    return (altitude, vertical_velocity, total_time, fuel_weight)


# (altitude, vertical_velocity, total_time, fuel_weight) = launch_rocket(45, 100)
# print(
#     f"The rocket reached altitude: {altitude:.2f}m in {total_time:.2f}s. Current speed {vertical_velocity:.2f}m/s and fuel-level: {fuel_weight:.2f}kg."
# )

# simulate_small_engine(N)
# plot_velocity_distribution(N)
# plot_small_engine()

"""Parametre"""
L = 10e-7  # Bredde på boksen i meter
T = 3000  # Gassens temperatur i kelvin
N = 10**4  # Antall partikler
t_c = 10e-9  # Tid
dt = 10e-12  # Tids intervall i s'
t = np.arange(0, t_c, dt)  # Tidsarray definert vha Tid og tidssteg
m_H2 = const.m_H2
k_B = const.k_B
MB = np.sqrt(const.k_B * T / const.m_H2)

engine1 = Engine(N=10**4, L=10e-8, n_A=5 * (10e-15), T=3000, t_c=10e-9, dt=10e-12)
engine2 = Engine(N=10**4, L=10e-7, n_A=5 * (10e-15), T=3000, t_c=10e-9, dt=10e-12)

print(engine1.thrust / engine1.total_fuel_constant)
print(engine2.thrust / engine2.total_fuel_constant)
