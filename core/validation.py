import math


def compare_theory_simulation(theory, simulation, tolerance=0.10):
    metrics = {}
    for key_theory, key_sim, label in [
        ("Wq", "Wq_hours", "Wq"),
        ("W", "W_hours", "W"),
        ("rho", "utilization", "Utilisation"),
    ]:
        t = theory.get(key_theory)
        s = simulation.get(key_sim)
        if t is None or s is None:
            continue

        denom = max(abs(t), 1e-9)
        relative_error = abs(s - t) / denom

        metrics[label] = {
            "theoretical": t,
            "simulated": s,
            "relative_error": relative_error,
            "within_tolerance": relative_error <= tolerance,
        }

    return metrics


def validate_little_law(lam, L, W, tolerance=0.02):
    expected = lam * W
    relative_error = abs(L - expected) / max(abs(expected), 1e-9)
    return {
        "expected_L": expected,
        "observed_L": L,
        "relative_error": relative_error,
        "valid": relative_error <= tolerance,
    }
