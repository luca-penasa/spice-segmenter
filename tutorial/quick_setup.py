import tempfile
from pathlib import Path

import numpy
from planetary_coverage import TourConfig

kdir = Path(tempfile.gettempdir()).joinpath("kernels")

config = TourConfig(
    kernels_dir=kdir,
    target="Jupiter",
    spacecraft="Juice",
    version="latest",
    mk="plan",
    download_kernels=True,
)

start, end = (
    numpy.datetime64("2023-04-14T12:42:17.657"),
    numpy.datetime64("2031-10-05T01:57:49.500"),
)

traj = config[start:end:"1 day"]
config.load_kernels()
