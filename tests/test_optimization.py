import math

from core.optimization import staffing_cost_optimization


def test_staffing_optimisation_returns_feasible_recommendation():
    result = staffing_cost_optimization(
        lam=12,
        mu=15,
        max_servers=5,
        server_cost_per_hour=500,
        waiting_cost_per_customer_hour=300,
    )

    table = result["table"]
    recommended = result["recommended_servers"]

    assert recommended in table["servers"].tolist()
    assert math.isfinite(result["minimum_total_cost"])
    assert table.loc[
        table["servers"] == recommended, "stable"
    ].iloc[0]


def test_staffing_optimisation_minimum_matches_recommended_row():
    result = staffing_cost_optimization(
        lam=12,
        mu=15,
        max_servers=5,
        server_cost_per_hour=500,
        waiting_cost_per_customer_hour=300,
    )

    table = result["table"]
    row = table[table["servers"] == result["recommended_servers"]].iloc[0]

    assert row["total_cost"] == result["minimum_total_cost"]


def test_staffing_optimisation_marks_unstable_server_counts():
    result = staffing_cost_optimization(
        lam=20,
        mu=15,
        max_servers=3,
    )

    table = result["table"]
    one_server = table[table["servers"] == 1].iloc[0]
    two_servers = table[table["servers"] == 2].iloc[0]

    assert not one_server["stable"]
    assert math.isinf(one_server["total_cost"])
    assert two_servers["stable"]
    assert result["recommended_servers"] is not None


def test_staffing_optimisation_no_feasible_servers():
    result = staffing_cost_optimization(
        lam=20,
        mu=15,
        max_servers=1,
    )

    assert result["recommended_servers"] is None
    assert result["minimum_total_cost"] is None
