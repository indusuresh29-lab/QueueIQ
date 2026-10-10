import math
import numpy as np
from typing import Dict


def _require_positive(lam: float, mu: float):
    if lam < 0 or mu <= 0:
        raise ValueError("Arrival rate must be >= 0 and service rate must be > 0.")


def mm1(lam: float, mu: float) -> Dict:
    """M/M/1 queue. Rates must use the same time unit."""
    _require_positive(lam, mu)
    rho = lam / mu

    if rho >= 1:
        return {
            "model": "M/M/1",
            "stable": False,
            "rho": rho,
            "message": "System is unstable because λ ≥ μ.",
        }

    Lq = rho**2 / (1 - rho)
    L = rho / (1 - rho)
    Wq = Lq / lam if lam > 0 else 0.0
    W = L / lam if lam > 0 else 1 / mu
    P_wait = rho

    return {
        "model": "M/M/1",
        "stable": True,
        "lambda": lam,
        "mu": mu,
        "servers": 1,
        "rho": rho,
        "Lq": Lq,
        "L": L,
        "Wq": Wq,
        "W": W,
        "P_wait": P_wait,
        "P0": 1 - rho,
        "capacity_ratio": rho,
    }


def md1(lam: float, mu: float) -> Dict:
    """M/D/1 queue using Pollaczek-Khinchine for deterministic service."""
    _require_positive(lam, mu)
    rho = lam / mu

    if rho >= 1:
        return {
            "model": "M/D/1",
            "stable": False,
            "rho": rho,
            "message": "System is unstable because λ ≥ μ.",
        }

    Wq = rho / (2 * mu * (1 - rho))
    Lq = lam * Wq
    W = Wq + 1 / mu
    L = lam * W

    return {
        "model": "M/D/1",
        "stable": True,
        "lambda": lam,
        "mu": mu,
        "servers": 1,
        "rho": rho,
        "Lq": Lq,
        "L": L,
        "Wq": Wq,
        "W": W,
        "P_wait": rho,
    }


def mmc(lam: float, mu: float, c: int) -> Dict:
    """M/M/c queue using Erlang-C."""
    _require_positive(lam, mu)
    if c < 1:
        raise ValueError("Number of servers must be at least 1.")

    rho = lam / (c * mu)

    if rho >= 1:
        return {
            "model": f"M/M/{c}",
            "stable": False,
            "rho": rho,
            "servers": c,
            "message": "System is unstable because λ ≥ cμ.",
        }

    a = lam / mu

    p0_denom = sum((a**n) / math.factorial(n) for n in range(c))
    p0_denom += (a**c) / (math.factorial(c) * (1 - rho))
    P0 = 1 / p0_denom

    P_wait = ((a**c) / (math.factorial(c) * (1 - rho))) * P0
    Lq = P_wait * rho / (1 - rho)
    Wq = Lq / lam if lam > 0 else 0.0
    W = Wq + 1 / mu
    L = lam * W

    return {
        "model": f"M/M/{c}",
        "stable": True,
        "lambda": lam,
        "mu": mu,
        "servers": c,
        "rho": rho,
        "Lq": Lq,
        "L": L,
        "Wq": Wq,
        "W": W,
        "P_wait": P_wait,
        "P0": P0,
    }


def finite_queue(lam: float, mu: float, K: int, c: int = 1) -> Dict:
    """Finite-capacity M/M/c/K queue. K is total system capacity."""
    _require_positive(lam, mu)
    if K < c:
        raise ValueError("Total capacity K must be >= number of servers c.")

    a = lam / mu
    terms = []
    for n in range(K + 1):
        if n <= c:
            term = (a**n) / math.factorial(n)
        else:
            term = (a**n) / (math.factorial(c) * (c ** (n - c)))
        terms.append(term)

    P0 = 1 / sum(terms)
    probs = np.array([P0 * x for x in terms])
    p_block = probs[K]
    lambda_eff = lam * (1 - p_block)
    L = float(sum(n * probs[n] for n in range(K + 1)))
    Lq = max(0.0, L - lambda_eff / mu)
    Wq = Lq / lambda_eff if lambda_eff > 0 else 0.0
    W = Wq + 1 / mu

    return {
        "model": f"M/M/{c}/{K}",
        "stable": True,
        "lambda": lam,
        "mu": mu,
        "servers": c,
        "capacity": K,
        "P_block": p_block,
        "lambda_effective": lambda_eff,
        "L": L,
        "Lq": Lq,
        "W": W,
        "Wq": Wq,
        "rho": lambda_eff / (c * mu),
    }


def finite_population(lam: float, mu: float, N: int) -> Dict:
    """Finite-source machine-repair style model with N sources and one server."""
    _require_positive(lam, mu)
    if N < 1:
        raise ValueError("Population N must be at least 1.")

    # λ is interpreted as per-source attempt rate.
    weights = [1.0]
    for n in range(1, N + 1):
        weights.append(weights[-1] * ((N - (n - 1)) * lam) / (n * mu))

    P0 = 1 / sum(weights)
    probs = np.array([P0 * w for w in weights])
    L = float(sum(n * probs[n] for n in range(N + 1)))
    lambda_eff = lam * (N - L)
    W = L / lambda_eff if lambda_eff > 0 else 0.0
    Wq = max(0.0, W - 1 / mu)
    Lq = lambda_eff * Wq

    return {
        "model": "Finite Population M/M/1",
        "stable": True,
        "lambda_per_source": lam,
        "mu": mu,
        "population": N,
        "P0": P0,
        "L": L,
        "Lq": Lq,
        "W": W,
        "Wq": Wq,
        "lambda_effective": lambda_eff,
    }


def little_law(lam: float, W_hours: float, Wq_hours: float | None = None):
    result = {
        "L_from_little": lam * W_hours,
        "lambda": lam,
        "W": W_hours,
    }
    if Wq_hours is not None:
        result["Lq_from_little"] = lam * Wq_hours
        result["Wq"] = Wq_hours
    return result


def server_scenarios(lam: float, mu: float, max_servers: int = 8):
    rows = []
    for c in range(1, max_servers + 1):
        r = mmc(lam, mu, c)
        rows.append(r)
    return rows


def validate_littles_law(result: Dict, lam: float, tolerance: float = 1e-6):
    if not result.get("stable", False):
        return {"valid": False, "reason": "Unstable model."}

    L_check = lam * result["W"]
    Lq_check = lam * result["Wq"]
    return {
        "valid": abs(L_check - result["L"]) <= tolerance
        and abs(Lq_check - result["Lq"]) <= tolerance,
        "L_difference": L_check - result["L"],
        "Lq_difference": Lq_check - result["Lq"],
    }
