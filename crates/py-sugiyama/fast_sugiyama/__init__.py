from importlib.metadata import version

from . import layout
from .layout import Layouts
from .lib import from_edges

__all__ = ["from_edges", "layout", "Layouts"]
__version__ = version(__name__)
