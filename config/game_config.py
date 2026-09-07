import os
import sys
from pathlib import Path


def get_resource_path(relative_path: str) -> Path:
    """Resolves relative paths for both dev mode and PyInstaller bundles."""
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller onefile temporary extraction path
        base_path = Path(sys._MEIPASS)
    else:
        # Standard development root directory
        base_path = Path(__file__).resolve().parent.parent

    return base_path / relative_path