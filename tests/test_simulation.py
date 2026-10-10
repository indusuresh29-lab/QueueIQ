import pytest

from core.queue_models import mm1, mmc
from core.simulation import simulate_queue


def test_mm1_simulation_matches_theory():
    lam, mu = 12, 15

    theoretical = mm1(lam, mu)
    simulated = simulate_queue(
        arrival_rate=lam,
        service_rate=mu,
        servers=1,
        n_customers=120_000,
        seed=42,
    )

    assert theoretical["stable"]
    assert simulated["Wq_hours"] == pytest.approx(
        theoretical["Wq"], rel=0.12
    )
    assert simulated["W_hours"] == pytest.approx(
        theoretical["W"], rel=0.06
    )
    assert simulated["Lq"] == pytest.approx(
        theoretical["Lq"], rel=0.12
    )


def test_mmc_simulation_matches_erlang_c():
    lam, mu, servers = 12, 15, 2

    theoretical = mmc(lam, mu, servers)
    simulated = simulate_queue(
        arrival_rate=lam,
        service_rate=mu,
        servers=servers,
        n_customers=120_000,
        seed=42,
    )

    assert theoretical["stable"]
    assert simulated["Wq_hours"] == pytest.approx(
        theoretical["Wq"], rel=0.15, abs=0.001
    )
    assert simulated["W_hours"] == pytest.approx(
        theoretical["W"], rel=0.04
    )


def test_simulation_is_reproducible_with_fixed_seed():
    first = simulate_queue(
        arrival_rate=8,
        service_rate=12,
        servers=1,
        n_customers=2_000,
        seed=123,
    )
    second = simulate_queue(
        arrival_rate=8,
        service_rate=12,
        servers=1,
        n_customers=2_000,
        seed=123,
    )

    assert first == second


@pytest.mark.parametrize(
    "arrival_rate, service_rate, servers",
    [
        (0, 10, 1),
        (5, 0, 1),
        (5, 10, 0),
    ],
)
def test_simulation_rejects_invalid_inputs(
    arrival_rate, service_rate, servers
):
    with pytest.raises(ValueError):
        simulate_queue(
            arrival_rate=arrival_rate,
            service_rate=service_rate,
            servers=servers,
            n_customers=100,
        )
