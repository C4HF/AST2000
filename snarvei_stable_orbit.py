from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts

seed = 57063
code_stable_orbit = 75980

mission = SpaceMission(seed)
shortcut = SpaceMissionShortcuts(mission, [code_stable_orbit])

################################################################
#               PLACE SPACECRAFT IN STABLE ORBIT               #
################################################################
#                   |      For Part 6      |
#                   ------------------------

"""
DOCUMENTATION

------------------------------------------------------------------------
place_spacecraft_in_stable_orbit() places the spacecraft in a circular
orbit around the specified planet.

Parameters
----------

time  :  float
    The time at which the spacecraft should be placed in orbit, in YEARS
    from the initial system time.

orbital_height  :  float
    The height of the orbit above the planet surface, in METERS.

orbital_angle  :  float
    The angle of the initial position of the spacecraft in orbit, in
    RADIANS relative to the x-axis.

planet_idx  :  int
    The index of the planet that the spacecraft should orbit.

Raises
------

RuntimeError
    When none of the provided codes are valid for unlocking this method.
RuntimeError
    When called before verify_manual_orientation() has been called
    successfully.
------------------------------------------------------------------------

"""

time = # insert the time you want the spacecraft to be placed in orbit
orbital_height = # insert the height of the orbit above the surface
orbital_angle = # insert the angle of initial position of spacecraft here
planet_idx = # insert the index of your destination planet

shortcut.place_spacecraft_in_stable_orbit(time, orbital_height,
    orbital_angle, planet_idx)

# initiating landing sequence. Documentation on how to use your
# LandingSequence instance can be found here:
#     https://lars-frogner.github.io/ast2000tools/html/classes/ast2000to
#     ols.space_mission.LandingSequence.html#ast2000tools.space_mission.
#     LandingSequence

land = mission.begin_landing_sequence()
# print()
# land.orient()
# print()
