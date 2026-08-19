"""ACN-session calibration for the synthetic simulator."""

from pathlib import Path

import pytest

from chargeopt.data.io import read_sessions_csv
from chargeopt.simulation.calibration import calibrate_from_sessions

FIXTURE = Path("tests/fixtures/acn_sessions.csv")


def test_fixture_calibration_preserves_arrival_energy_and_duration_statistics() -> None:
    sessions = read_sessions_csv(FIXTURE)

    calibration = calibrate_from_sessions(sessions, timestep_minutes=15)

    assert len(calibration.hour_pmf) == 24
    assert sum(calibration.hour_pmf) == pytest.approx(1.0)
    assert calibration.hour_pmf[8] == max(calibration.hour_pmf)
    assert calibration.mean_duration_min == pytest.approx(105.0)
    assert calibration.mean_energy_kwh == pytest.approx(71.0 / 6.0)
    assert calibration.peak_concurrent >= 1
    assert calibration.n_sessions == 6


def test_calibration_rejects_empty_sessions() -> None:
    sessions = read_sessions_csv(FIXTURE).iloc[0:0]

    with pytest.raises(ValueError, match="empty"):
        calibrate_from_sessions(sessions, timestep_minutes=15)
