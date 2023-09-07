########## Egen kode #############################
import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem
from scipy.stats import norm

seed = 57063
system = SolarSystem(seed)
mission = SpaceMission(seed)
utils.check_for_newer_version()
# @jit(nopython = True) #Optimalisering(?)

"""Parametre"""
m_H2 = const.m_H2
k_B = const.k_B
SM = 1.9891 * 10 ** (30)  # Solar masses in kg
G = 6.6743 * (10 ** (-11))  # Gravitational constant
dry_rocket_mass = mission.spacecraft_mass
crosssection_rocket = mission.spacecraft_area
homeplanet_radius = system._radii[0] * 1000  # homeplanet radius in m
homeplanet_mass = system._masses[0] * SM  # homeplanet mass in kg
home_planet_initial_vel = (
    system._initial_velocities[0][0],
    system._initial_velocities[1][0],
)  # homeplanet initital velocity
home_planet_initial_pos = (
    system._initial_positions[0][0],
    system._initial_positions[1][0],
)  # homeplanet initial pos
home_planet_rotational_period = system._rotational_periods[
    0
]  # homeplanet rotational period

escape_velocity = np.sqrt((2 * G * homeplanet_mass) / homeplanet_radius)
print(system.__dir__())  ### get a list of attribute-commands
# print(mission.__dir__())
print(homeplanet_radius)
print(homeplanet_mass)
print(home_planet_initial_vel)
print(home_planet_initial_pos)
print(home_planet_rotational_period)


"""Kode for 1B og 1C."""


class Engine:
    """Create instance of engine with N particles, L dimension small engine,
    n_A nozzle area in percent, T temperature in Kelvin, t_c simulation time, dt simulation timestep.
    """

    def __init__(self, N: int, L: float, n_A: float, T: float, t_c: float, dt: float):
        if n_A < 0 or n_A > 1:
            raise ValueError(f"n_A should be between 0 and 1. Got n_A = {n_A:.2f}.")
        self.N = N
        self.L = L
        self.n_A = n_A
        self.T = T
        self.t_c = t_c
        self.dt = dt
        self._simulate_small_engine()  # function is called by constructer

    def _simulate_small_engine(self):
        """Simulate small engine performance. Stores results as class-attributes."""
        N = self.N
        L = self.L
        n_A = self.n_A
        T = self.T
        t_c = self.t_c
        dt = self.dt
        t = np.arange(0, t_c, dt)
        l = L * np.sqrt(n_A)
        d = (L - l) / 2
        m_H2 = const.m_H2
        k_B = const.k_B
        self.MB = np.sqrt(const.k_B * T / const.m_H2)

        a = []  # Skal telle farten til partiklene som slipper ut
        rows = 3  # For vectors
        cols = N
        pos = L * np.random.rand(rows, cols)  # Particle positions
        mean = 0
        std = self.MB
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
                if d < pos[0][z2[m]] < (L - d) and d < pos[1][z2[m]] < (
                    L - d
                ):  # Sjekker om kollisjonene for z2(xy-planet) egentlig er i utgangshullet
                    if d < pos[1][z2[m]] < (L - d):
                        a.append(
                            vel[2][z2[m]]
                        )  # Lagrer farten til partiklene som forsvinner ut. Kan brukes til beregninger
                        for i in range(
                            2
                        ):  # Flytter partikkelen til en uniformt fordelt posisjon på toppen av boksen, med samme vel.
                            pos[i][z2[m]] = L * np.random.rand()
                        pos[2][z2[m]] = L * 0.99
                        vel[2][z2[m]] = -vel[2][z2[m]]

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
        self.vel = vel

        # Trykk
        self.simulated_average_pressure = sum(pressure_list) / len(pressure_list)
        n = N / (L * L * L)
        self.analytical_expected_pressure = n * k_B * T

        # Energi
        self.simulated_kinetic_energy = 1 / 2 * m_H2 * np.power(vel, 2)
        self.simulated_total_energy = np.sum(self.simulated_kinetic_energy)
        self.simulated_average_energy = self.simulated_total_energy / N
        self.analytical_expected_energy = (3 / 2) * k_B * T

        # Fuel consumption
        self.tot_fuel = m_H2 * len(a)
        self.fuel_cons = self.tot_fuel / t_c

        # Fremdrift
        self.P = sum(a) * m_H2  # P = mv, bruker bare v_z da de andre blir 0 totalt.
        self.F = -self.P / t_c  # F = mv / dt

        ## Utregning av total thrust og total fuel-constant
        self.number_of_engines = crosssection_rocket * 10 / (L**2)
        self.number_of_engines = crosssection_rocket * 10 / (L**2)
        self.thrust = self.number_of_engines * self.F
        self.total_fuel_constant = self.fuel_cons * self.number_of_engines

    def plot_velocity_distribution(self, bins=30):
        """Funksjonen kaller på simulate_small_engine funksjonen og danner et
        subplot som sammenligner den simulerte farten med Maxwell-Boltzmann fordelingen.
        Tar inn npb variabel (antall partikler per boks), bins (antall stolper i histogrammet).
        Returnerer ingenting"""
        vel = self.vel
        N = self.N
        mean = 0
        std = self.MB
        x_axis = np.linspace(-4 * std, 4 * std, N)
        bins = bins
        # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True)

        plt.hist(
            vel[0], bins=bins, alpha=0.4, density=True, label="x-velocity", color="cyan"
        )
        plt.plot(
            x_axis,
            norm.pdf(x_axis, loc=mean, scale=std),
            color="red",
            label="MB-dist",
        )

        plt.hist(
            vel[1],
            bins=bins,
            alpha=0.4,
            density=True,
            label="y-velocity",
            color="olive",
        )
        # ax2.plot(
        #     x_axis,
        #     norm.pdf(x_axis, loc=mean, scale=std),
        #     color="red",
        #     label="MB-dist",
        # )

        plt.hist(
            vel[2], bins=bins, alpha=0.4, density=True, label="z-velocity", color="pink"
        )
        # ax3.plot(
        #     x_axis,
        #     norm.pdf(x_axis, loc=mean, scale=std),
        #     color="red",
        #     label="MB-dist",
        # )
        plt.ylabel("%", fontsize=25)
        plt.xlabel("m/s", fontsize=25)
        plt.xticks(fontsize=23)
        plt.yticks(fontsize=23)
        plt.legend(loc="upper left", fontsize=25)
        # ax2.set_xlabel("m/s")
        # ax2.set_title("Velocity y-direction")
        # ax2.legend(loc="upper left")
        # ax3.set_xlabel("m/s")
        # ax3.set_title("Velocity z-direction")
        # ax3.legend(loc="upper left")
        plt.suptitle(
            "Simulated velocity of our particles compared to Maxwell-Boltzmann",
            fontsize=25,
        )
        plt.show()

    def plot_small_engine(self, npb=10, time=100):
        """Plots a simulation of particle movement inside one small engine, with fewer particles."""
        from mpl_toolkits import mplot3d  # Plotting

        n_A = self.n_A
        L = self.L
        MB = self.MB
        l = L * np.sqrt(n_A)
        d = (L - l) / 2
        dt = self.dt

        fig = plt.figure()
        ax = plt.axes(projection="3d")  # For 3d plotting av rakettmotoren
        ax.plot3D(
            [0, L], [0, 0], [0, 0], "green"
        )  # Lager en yttre firkant på xy-planet
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
                if d < pos[0][z2[m]] < (L - d):  # egentlig er i
                    if d < pos[1][z2[m]] < (L - d):  # utgangshullet
                        for i in range(2):  # Flytter partikkelen til en uniformt
                            pos[i][m] = L * np.random.rand()  # fordelt posisjon på
                        pos[2][m] = L  # toppen av boksen, med samme vel.
                        if z2[m] not in nr:  # Brukes til plotting
                            nr.append(z2[m])
            z2 = list(z2)
            x1 = list(x1)
            x2 = list(x2)
            y1 = list(y1)
            y2 = list(y2)
            for i in range(len(nr)):  # For plotting
                if nr[i] in z2:  # Av at partiklene fyker ut av boksen
                    z2.remove(nr[i])
                if nr[i] in x1:
                    x1.remove(nr[i])
                if nr[i] in x2:
                    x2.remove(nr[i])
                if nr[i] in y1:
                    y1.remove(nr[i])
                if nr[i] in y2:
                    y2.remove(nr[i])

            vel[0][x1] = -vel[0][x1]
            vel[0][x2] = -vel[0][x2]  # Elastisk støt ved å snu
            vel[1][y1] = -vel[1][y1]  # farten til det motsatte i en gitt retning
            vel[1][y2] = -vel[1][y2]
            vel[2][z1] = -vel[2][z1]
            vel[2][z2] = -vel[2][z2]

            ax.scatter(pos[0], pos[1], pos[2])  # Plotter rakettmotoren

        plt.show()


"""Kode 1D"""


def calculate_needed_fuel(engine, initial_rocket_mass, speed_boost, dt=10):
    """Funksjonen regner ut hvor mye drivstoff vi trenger for å øke hastigheten med en ønsket mengde.
    Tar inn variabler (thrust, fuel_consumption, initial_rocket_mass, speed_boost) og returnerer
    fuel_consumed."""
    thrust = engine.thrust
    fuel_consumption = engine.fuel_cons
    start_speed = 0
    total_time = 0
    while start_speed < speed_boost:
        start_speed += (thrust / initial_rocket_mass) * dt
        initial_rocket_mass -= fuel_consumption * dt
        total_time += dt
    fuel_consumed = total_time * fuel_consumption
    return fuel_consumed


"""Kode 1E"""


def launch_rocket(engine, fuel_weight, target_vertical_velocity, dt=10):
    """Funksjonen simulerer launch av raketten."""
    thrust = engine.thrust
    total_fuel_constant = engine.total_fuel_constant

    altitude = 0
    vertical_velocity = 0
    total_time = 0

    while vertical_velocity < target_vertical_velocity:
        wet_rocket_mass = dry_rocket_mass + fuel_weight
        F_g = (G * homeplanet_mass * wet_rocket_mass) / (
            homeplanet_radius + altitude
        ) ** 2  # The gravitational force
        rocket_thrust_gravitation_diff = thrust - F_g
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


# def get_solar_navigation(engine):


### Eksempel på bruk av engine-class: ########
falcon_engine = Engine(
    N=2 * 10**4, L=3.775 * 10e-8, n_A=1, T=3300, t_c=10e-11, dt=10e-14
)
falcon_engine.plot_velocity_distribution()
# print((G * homeplanet_mass * 5000) / ((homeplanet_radius**2) * k_B * 3000 * (16)))
# L = ((10**5) / (2 * 10**24)) ** (1 / 3)
# print(L)

# print(calculate_needed_fuel(falcon_engine, 30000, 59757))
print(launch_rocket(falcon_engine, 165000, escape_velocity, dt=1))
print(f"Thrust: {falcon_engine.thrust}, {falcon_engine.thrust / 165000}")
print(f"Total fuel constant: {falcon_engine.total_fuel_constant}")
print(
    f"Thrust/total fuel constant: {falcon_engine.thrust / falcon_engine.total_fuel_constant}"
)
print(f"Simulated pressure: {falcon_engine.simulated_average_pressure}")
print(f"Expected pressure: {falcon_engine.analytical_expected_pressure}")
print(f"Simulated total energy: {falcon_engine.simulated_total_energy}")
print(f"Simulated average energy: {falcon_engine.simulated_average_energy}")
print(f"Analytical expected energy: {falcon_engine.analytical_expected_energy}")

plt.show()
