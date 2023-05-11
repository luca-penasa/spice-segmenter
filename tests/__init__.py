from functools import partial
from pathlib import Path
from tempfile import gettempdir

from planetary_coverage import TourConfig as _TourConfig

tmp_kernels_dir = Path(gettempdir()) / "kernels"


TourConfig = partial(
    _TourConfig,
    kernels_dir=tmp_kernels_dir,
    download_kernels=True,
    target="JUPITER",
    spacecraft="JUICE",
    load_kernels=True,
    version="v432_20230505_001",
    mk="plan",
)
