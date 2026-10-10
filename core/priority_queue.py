import math
from typing import List, Dict


def nonpreemptive_priority_m_m_1(classes: List[Dict]) -> Dict:
    """
    Approximate analytical non-preemptive M/M/1 priority results.

    Each class requires:
      name, priority (1 = highest), lambda, mu

    Formula:
      W0 = sum(lambda_i / mu_i^2)
      Wq_i = W0 / [(1 - cumulative_rho_i)(1 - cumulative_rho_before_i)]
    """
    if not classes:
        raise ValueError("At least one priority class is required.")

    ordered = sorted(classes, key=lambda x: x["priority"])
    total_rho = sum(x["lambda"] / x["mu"] for x in ordered)

    if total_rho >= 1:
        return {
            "stable": False,
            "total_rho": total_rho,
            "message": "Total offered load is >= 1; priority system is unstable."
        }

    W0 = sum(x["lambda"] / (x["mu"] ** 2) for x in ordered)
    cumulative_before = 0.0
    results = []

    for item in ordered:
        rho_i = item["lambda"] / item["mu"]
        cumulative_i = cumulative_before + rho_i

        Wq = W0 / ((1 - cumulative_i) * (1 - cumulative_before))
        W = Wq + 1 / item["mu"]
        Lq = item["lambda"] * Wq
        L = item["lambda"] * W

        results.append({
            "name": item["name"],
            "priority": item["priority"],
            "lambda": item["lambda"],
            "mu": item["mu"],
            "rho": rho_i,
            "Wq": Wq,
            "W": W,
            "Lq": Lq,
            "L": L,
        })
        cumulative_before = cumulative_i

    weighted_Wq = (
        sum(x["lambda"] * x["Wq"] for x in results) / sum(x["lambda"] for x in results)
    )

    return {
        "stable": True,
        "total_rho": total_rho,
        "W0": W0,
        "classes": results,
        "weighted_average_Wq": weighted_Wq,
        "discipline": "Non-preemptive priority",
    }
