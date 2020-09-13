# type: ignore[attr-defined]
"""PyPolSAR is a python module for Polarimetric Synthetic Aperture Radar (PolSAR) data processing."""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError


try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
