import tempfile
from pathlib import Path

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

start, end = config.coverage

traj = config[start:end:"1 day"]
config.load_kernels()
