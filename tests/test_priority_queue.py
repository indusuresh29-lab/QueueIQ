import pytest

from core.priority_queue import nonpreemptive_priority_m_m_1
from core.simulation import simulate_priority_queue


def sample_classes():
    return [
        {
            "name": "High",
            "priority": 1,
            "lambda": 2,
            "mu": 10,
        },
        {
            "name": "Low",
            "priority": 2,
            "lambda": 3,
            "mu": 10,
        },
    ]


def test_priority_analytical_model_orders_classes():
    result = nonpreemptive_priority_m_m_1(sample_classes())

    assert result["stable"]
    assert result["discipline"] == "Non-preemptive priority"
    assert [item["name"] for item in result["classes"]] == [
        "High",
        "Low",
    ]


def test_priority_analytical_waiting_metrics_are_consistent():
    result = nonpreemptive_priority_m_m_1(sample_classes())

    assert result["stable"]

    for item in result["classes"]:
        assert item["Wq"] >= 0
        assert item["W"] == pytest.approx(
            item["Wq"] + 1 / item["mu"]
        )
        assert item["Lq"] == pytest.approx(
            item["lambda"] * item["Wq"]
        )
        assert item["L"] == pytest.approx(
            item["lambda"] * item["W"]
        )

    assert (
        result["classes"][0]["Wq"]
        < result["classes"][1]["Wq"]
    )


def test_priority_analytical_model_rejects_empty_classes():
    with pytest.raises(ValueError):
        nonpreemptive_priority_m_m_1([])


def test_priority_analytical_model_flags_instability():
    classes = [
        {"name": "High", "priority": 1, "lambda": 6, "mu": 10},
        {"name": "Low", "priority": 2, "lambda": 5, "mu": 10},
    ]

    result = nonpreemptive_priority_m_m_1(classes)

    assert not result["stable"]
    assert result["total_rho"] >= 1


def test_priority_simulation_returns_class_level_results():
    result = simulate_priority_queue(
        classes=sample_classes(),
        servers=1,
        n_customers_per_class=500,
        seed=42,
    )

    assert not result.empty
    assert set(result["class"]) == {"High", "Low"}
    assert (result["customers"] > 0).all()
    assert (result["Wq_hours"] >= 0).all()
    assert (result["W_hours"] >= result["Wq_hours"]).all()


def test_priority_simulation_rejects_preemptive_mode():
    with pytest.raises(NotImplementedError):
        simulate_priority_queue(
            classes=sample_classes(),
            servers=1,
            n_customers_per_class=100,
            preemptive=True,
        )
