"Klipp og lim AST"
 # def simulate_small_engine(
    #     npb,
    # ):
    #     """Funksjonen simulerer bevegelsen til partiklene i en liten boks.
    #     Tar inn variabel npb (antall partikler per boks) og returnerer
    #     tuppel: (vel, average_pressure, average_energy, F, fuel_cons).
    #     """
    #     a = []  # Skal telle farten til partiklene som slipper ut
    #     rows = 3  # For vectors
    #     cols = npb
    #     pos = L * np.random.rand(rows, cols)  # Particle positions
    #     mean = 0
    #     std = MB
    #     vel = np.random.normal(
    #         loc=mean, scale=std, size=(rows, cols)
    #     )  # loc = mean, scale = standard deviation(std)
    #     for i in range(
    #         len(vel)
    #     ):  # Sørger for at ingen hastigheter er negative(Sjelden feil)
    #         vel_0 = np.where(vel[i] == 0)[0]
    #         vel[i][vel_0] = vel[i - 1][vel_0]

    #     pressure_list = []

    #     for m in range(len(t)):  # tidssteg
    #         pos += dt * vel  # Euler cromer

    #         x1 = np.where(pos[0] >= L)[0]  # Ser etter kollisjoner for x
    #         x2 = np.where(pos[0] <= 0)[0]
    #         y1 = np.where(pos[1] >= L)[0]  # Ser etter kollisjoner for y
    #         y2 = np.where(pos[1] <= 0)[0]
    #         z1 = np.where(pos[2] >= L)[0]  # Ser etter kollisjoner for z
    #         z2 = np.where(pos[2] <= 0)[0]

    #         for m in range(len(z2)):
    #             if (
    #                 L / 8 < pos[0][z2[m]] < (7 / 8) * L
    #             ):  # Sjekker om kollisjonene for z2(xy-planet) egentlig er i utgangshullet
    #                 if L / 8 < pos[1][z2[m]] < (7 / 8) * L:
    #                     a.append(
    #                         vel[2][z2[m]]
    #                     )  # Lagrer farten til partiklene som forsvinner ut. Kan brukes til beregninger
    #                     for i in range(
    #                         2
    #                     ):  # Flytter partikkelen til en uniformt fordelt posisjon på toppen av boksen, med samme vel.
    #                         pos[i][m] = L * np.random.rand()
    #                     pos[2][m] = L

    #         vel[0][x1] = -vel[0][x1]
    #         vel[0][x2] = -vel[0][
    #             x2
    #         ]  # Elastisk støt ved å snu farten til det motsatte i en gitt retning
    #         vel[1][y1] = -vel[1][y1]
    #         vel[1][y2] = -vel[1][y2]
    #         vel[2][z1] = -vel[2][z1]
    #         vel[2][z2] = -vel[2][z2]

    #         # ax.scatter(pos[0], pos[1], pos[2])  # Plotter rakettmotoren

    #         # Trykk per tidsteg
    #         momentum = vel * m_H2
    #         df = np.sum(
    #             (2 * np.abs(momentum[0][x1])) / dt
    #         )  # regner ut kraft som virker på veggen per tidssteg
    #         dp = df / (L * L)
    #         pressure_list.append(dp)

    #     # Trykk
    #     average_pressure = sum(pressure_list) / len(pressure_list)
    #     n = N / (L * L * L)
    #     analytical_pressure = n * k_B * T

    #     # Energi
    #     numerical_kinetic_energy = 1 / 2 * m_H2 * vel**2
    #     numerical_total_energy = np.sum(numerical_kinetic_energy)
    #     average_energy = numerical_total_energy / N
    #     analytical_average_energy = (3 / 2) * k_B * T

    #     # Fuel consumption
    #     tot_fuel = m_H2 * len(a)
    #     fuel_cons = tot_fuel / t_c

    #     # Fremdrift
    #     P = sum(a) * m_H2  # P = mv, bruker bare v_z da de andre blir 0 totalt.
    #     F = -P / t_c  # F = mv / dt

    #     return vel, average_pressure, average_energy, F, fuel_cons

    # def plot_velocity_distribution(npb, bins=30):
    #     """Funksjonen kaller på simulate_small_engine funksjonen og danner et
    #     subplot som sammenligner den simulerte farten med Maxwell-Boltzmann fordelingen.
    #     Tar inn npb variabel (antall partikler per boks), bins (antall stolper i histogrammet).
    #     Returnerer ingenting"""
    #     (vel, average_pressure, average_energy, F, fuel_cons) = simulate_small_engine(npb)
    #     mean = 0
    #     std = MB
    #     x_axis = np.linspace(-4 * std, 4 * std, N)
    #     bins = bins
    #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

    #     ax1.hist(
    #         vel[0], bins=bins, alpha=0.4, density=True, label="x-velocity", color="cyan"
    #     )
    #     ax1.plot(
    #         x_axis,
    #         norm.pdf(x_axis, loc=mean, scale=std),
    #         color="red",
    #         label="MB-dist",
    #     )

    #     ax2.hist(
    #         vel[1], bins=bins, alpha=0.4, density=True, label="y-velocity", color="olive"
    #     )
    #     ax2.plot(
    #         x_axis,
    #         norm.pdf(x_axis, loc=mean, scale=std),
    #         color="red",
    #         label="MB-dist",
    #     )

    #     ax3.hist(
    #         vel[2], bins=bins, alpha=0.4, density=True, label="z-velocity", color="pink"
    #     )
    #     ax3.plot(
    #         x_axis,
    #         norm.pdf(x_axis, loc=mean, scale=std),
    #         color="red",
    #         label="MB-dist",
    #     )
    #     ax1.set_ylabel("%")
    #     ax1.set_xlabel("m/s")
    #     ax1.set_title("Velocity x-direction")
    #     ax1.legend(loc="upper left")
    #     ax2.set_xlabel("m/s")
    #     ax2.set_title("Velocity y-direction")
    #     ax2.legend(loc="upper left")
    #     ax3.set_xlabel("m/s")
    #     ax3.set_title("Velocity z-direction")
    #     ax3.legend(loc="upper left")
    #     fig.suptitle("Simulated velocity of our particles compared to Maxwell-Boltzmann")
    #     plt.show()

    # def plot_small_engine(npb=100, time=100):