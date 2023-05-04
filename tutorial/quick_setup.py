import tempfile
from pathlib import Path

from planetary_coverage import TourConfig

kdir = Path(tempfile.tempdir).joinpath("kernels")

config = TourConfig(
    kernels_dir=kdir,
    target="Jupiter",
    spacecraft="Juice",
    version="latest",
    mk="plan",
    download_kernels=True,
)
traj = config[slice(*config.coverage, "1 day")]
config.load_kernels()
