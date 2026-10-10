import pandas as pd
import pytest

from core.sensitivity import (
    sensitivity_lambda,
    sensitivity_mu,
    server_sensitivity,
)


def test_lambda_sensitivity_returns_expected_scenarios():
    result = sensitivity_lambda(
        12, 15, pct_changes=[-0.1, 0, 0.1], servers=1
    )

    assert len(result) == 3
    assert result["lambda"].tolist() == pytest.approx(
        [10.8, 12, 13.2]
    )
    assert result["stable"].all()


def test_increasing_arrivals_increases_m_m_1_waiting_time():
    result = sensitivity_lambda(
        12, 15, pct_changes=[0, 0.05, 0.1], servers=1
    )

    waits = result["Wq_hours"].tolist()
    assert waits[0] < waits[1] < waits[2]


def test_service_rate_sensitivity_reflects_capacity_changes():
    result = sensitivity_mu(
        12, 15, pct_changes=[-0.1, 0, 0.1], servers=1
    )

    assert result["stable"].all()
    waits = result["Wq_hours"].tolist()
    assert waits[0] > waits[1] > waits[2]


def test_lambda_sensitivity_marks_unstable_scenarios():
    result = sensitivity_lambda(
        12, 15, pct_changes=[0, 0.3], servers=1
    )

    assert bool(result.iloc[0]["stable"])
    assert not bool(result.iloc[1]["stable"])
    assert pd.isna(result.iloc[1]["Wq_hours"])


def test_server_sensitivity_shows_improving_capacity():
    result = server_sensitivity(12, 15, max_servers=4)

    assert result["servers"].tolist() == [1, 2, 3, 4]
    assert result["stable"].all()
    waits = result["Wq_hours"].tolist()
    assert waits[0] > waits[1] > waits[2] > waits[3]
