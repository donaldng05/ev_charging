"""EV fleet charging intelligence."""

from importlib.metadata import PackageNotFoundError, version

from chargeopt.config import AppConfig, load_config

try:
    __version__ = version("chargeopt")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = ["AppConfig", "__version__", "load_config"]
