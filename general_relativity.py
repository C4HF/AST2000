from ast2000tools.relativity import RelativityExperiments

seed = 57063
experiments = RelativityExperiments(seed)


planet_idx = 1  # I want to perform the experiment near planet 1


experiments.spaceship_duel(
    planet_idx,
    filename_1="spaceship_duel_frame_1.xml",
    filename_2="spaceship_duel_frame_2.xml",
)

experiments.cosmic_pingpong(
    planet_idx,
    filename_1="cosmic_pingpong_frame_1.xml",
    filename_2="cosmic_pingpong_frame_2.xml",
    filename_3="cosmic_pingpong_frame_3.xml",
)
