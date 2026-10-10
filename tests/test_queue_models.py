import pytest

from core.queue_models import mm1, mmc, md1, finite_queue, finite_population, little_law, validate_littles_law


def test_mm1():
    result = mm1(12, 15)
    assert result["stable"]
    assert abs(result["rho"] - 0.8) < 1e-9
    assert abs(result["Lq"] - 3.2) < 1e-9
    assert abs(result["Wq"] - (3.2 / 12)) < 1e-9


def test_mmc_stability():
    result = mmc(12, 15, 2)
    assert result["stable"]
    assert result["rho"] == 0.4


def test_md1():
    result = md1(12, 15)
    assert result["stable"]
    assert result["rho"] == 0.8


def test_finite_queue():
    lam, mu, capacity = 8, 10, 6
    result = finite_queue(lam, mu, capacity)

    assert result["stable"]
    assert result["capacity"] == capacity
    assert 0 <= result["P_block"] <= 1
    assert result["lambda_effective"] == pytest.approx(
        lam * (1 - result["P_block"])
    )
    assert result["W"] == pytest.approx(
        result["Wq"] + 1 / mu
    )


def test_finite_population():
    result = finite_population(lam=0.5, mu=2, N=10)

    assert result["stable"]
    assert result["population"] == 10
    assert result["L"] >= 0
    assert result["L"] <= 10
    assert result["lambda_effective"] >= 0
    assert result["W"] == pytest.approx(
        result["L"] / result["lambda_effective"]
    )
    assert result["Wq"] >= 0
    assert result["Lq"] == pytest.approx(
        result["lambda_effective"] * result["Wq"]
    )


def test_little_law_calculations():
    result = little_law(
        lam=12,
        W_hours=1 / 3,
        Wq_hours=4 / 15,
    )

    assert result["L_from_little"] == pytest.approx(4)
    assert result["Lq_from_little"] == pytest.approx(3.2)


def test_littles_law_validation_for_mm1():
    result = mm1(12, 15)
    validation = validate_littles_law(result, lam=12)

    assert validation["valid"]
    assert abs(validation["L_difference"]) < 1e-6
    assert abs(validation["Lq_difference"]) < 1e-6


def test_littles_law_rejects_unstable_model():
    result = mm1(15, 15)
    validation = validate_littles_law(result, lam=15)

    assert not validation["valid"]
    assert validation["reason"] == "Unstable model."


@pytest.mark.parametrize(
    "lam, mu",
    [
        (15, 15),
        (18, 15),
    ],
)
def test_mm1_rejects_unstable_steady_state(lam, mu):
    result = mm1(lam, mu)

    assert not result["stable"]
    assert result["rho"] >= 1


def test_mm1_near_capacity_remains_stable():
    result = mm1(14.9, 15)

    assert result["stable"]
    assert result["rho"] < 1
    assert result["Lq"] > 100
    assert result["Wq"] > 0


def test_mmc_rejects_unstable_steady_state():
    result = mmc(30, 15, 2)

    assert not result["stable"]
    assert result["rho"] == 1


def test_mmc_near_capacity_remains_stable():
    result = mmc(29.9, 15, 2)

    assert result["stable"]
    assert result["rho"] < 1
    assert result["Lq"] > 100
