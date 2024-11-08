import numpy
from quick_spice_manager import SpiceManager

config = SpiceManager().tour_config

start, end = (
    numpy.datetime64("2023-04-14T12:42:17.657"),
    numpy.datetime64("2031-10-05T01:57:49.500"),
)

traj = config[start:end:"1 day"]
config.load_kernels()
