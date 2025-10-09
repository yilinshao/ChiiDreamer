import threestudio
from packaging.version import Version

if hasattr(threestudio, "__version__") and Version(threestudio.__version__) >= Version(
    "0.2.1"
):
    pass
else:
    if hasattr(threestudio, "__version__"):
        print(f"[INFO] threestudio version: {threestudio.__version__}")
    raise ValueError(
        "threestudio version must be >= 0.2.0, please update threestudio by pulling the latest version from github"
    )

from .data import random_multiview
from .background import gaussian_mvdream_background
from .geometry import exporter, gaussian_base, gaussian_io
import sys
print(sys.modules.keys())
from .utils import np_utils
print(sys.modules.keys())
from .material import gaussian_material
from .guidance import spatial_guidance, sd_latent
from .renderer import (
    diff_gaussian_rasterizer,
    diff_gaussian_rasterizer_advanced,
    diff_gaussian_rasterizer_background,
    diff_gaussian_rasterizer_shading,
)
from .system import gaussian_mvdream, gaussian_splatting, gaussian_zero123, layout_gaussian, fused_guidance_gs, chii_dreamer
from .utils import layout_utils

from .system import layout_gaussian
