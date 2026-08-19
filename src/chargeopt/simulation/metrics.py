"""Aggregate one simulation run into the frozen MVP metric contract."""

from chargeopt.simulation.schemas import SimMetrics


def build_metrics(
    *,
    seed: int,
    energy_cost: float,
    queued_vehicle_ticks: int,
    charge_sessions: int,
    soc_violations: int,
    energy_usage_kwh: float,
    occupied_charger_ticks: int,
    available_charger_ticks: int,
    idle_vehicle_ticks: int,
    timestep_minutes: int,
) -> SimMetrics:
    """Convert engine counters to comparable run metrics."""
    wait_denominator = charge_sessions if charge_sessions else 1
    utilization_denominator = available_charger_ticks if available_charger_ticks else 1
    return SimMetrics(
        seed=seed,
        energy_cost=energy_cost,
        avg_wait_minutes=(queued_vehicle_ticks * timestep_minutes / wait_denominator),
        soc_violations=soc_violations,
        energy_usage_kwh=energy_usage_kwh,
        station_utilization=occupied_charger_ticks / utilization_denominator,
        vehicle_idle_minutes=idle_vehicle_ticks * timestep_minutes,
    )
