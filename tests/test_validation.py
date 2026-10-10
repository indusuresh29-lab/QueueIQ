from core.validation import (
    compare_theory_simulation,
    validate_little_law,
)


def test_theory_simulation_comparison_within_tolerance():
    theory = {"Wq": 0.2, "W": 0.3, "rho": 0.8}
    simulation = {
        "Wq_hours": 0.21,
        "W_hours": 0.31,
        "utilization": 0.82,
    }

    result = compare_theory_simulation(theory, simulation, tolerance=0.10)

    assert result["Wq"]["within_tolerance"]
    assert result["W"]["within_tolerance"]
    assert result["Utilisation"]["within_tolerance"]


def test_theory_simulation_comparison_flags_large_error():
    theory = {"Wq": 0.2}
    simulation = {"Wq_hours": 0.5}

    result = compare_theory_simulation(theory, simulation, tolerance=0.10)

    assert not result["Wq"]["within_tolerance"]


def test_theory_simulation_comparison_skips_missing_metrics():
    result = compare_theory_simulation(
        {"Wq": 0.2},
        {},
    )

    assert result == {}


def test_little_law_validation():
    result = validate_little_law(
        lam=12,
        L=4.0,
        W=1 / 3,
    )

    assert result["valid"]
    assert result["relative_error"] < 1e-9


def test_little_law_validation_flags_inconsistency():
    result = validate_little_law(
        lam=12,
        L=8.0,
        W=1 / 3,
        tolerance=0.02,
    )

    assert not result["valid"]
    assert result["relative_error"] > 0.02
