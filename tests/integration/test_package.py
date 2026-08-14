"""Package import smoke test."""

from chargeopt import __version__, load_config


def test_version_is_present() -> None:
    assert __version__


def test_load_config_from_package() -> None:
    config = load_config()
    assert config.simulation.fleet_size > 0
