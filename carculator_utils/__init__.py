"""

Submodules
==========

.. autosummary::
    :toctree: _autosummary


"""

__all__ = (
    "get_standard_driving_cycle_and_gradient",
    "NoiseEmissionsModel",
    "HotEmissionsModel",
    "Inventory",
    "BackgroundSystemModel",
    "ExportInventory",
    "VehicleInputParameters",
)
__version__ = (1, 0, 9)

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

from .background_systems import BackgroundSystemModel
from .driving_cycles import get_standard_driving_cycle_and_gradient
from .export import ExportInventory
from .hot_emissions import HotEmissionsModel
from .inventory import Inventory
from .noise_emissions import NoiseEmissionsModel
from .vehicle_input_parameters import VehicleInputParameters
