# clic/__init__.py
"""
CLIC: Contrastive Learning for Image Complexity
"""

from . import builder
from . import loader
from . import CAL

__version__ = "1.0.0"
__all__ = ["builder", "loader", "CAL"]