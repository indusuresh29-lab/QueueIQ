import pandas as pd
from .queue_models import mm1, mmc


def sensitivity_lambda(lam, mu, pct_changes=None, servers=1):
    if pct_changes is None:
        pct_changes = [-0.20, -0.10, 0.0, 0.10, 0.20]

    rows = []
    for change in pct_changes:
        new_lam = lam * (1 + change)
        result = mm1(new_lam, mu) if servers == 1 else mmc(new_lam, mu, servers)
        rows.append({
            "change_pct": change * 100,
            "lambda": new_lam,
            "rho": result.get("rho"),
            "Wq_hours": result.get("Wq"),
            "Lq": result.get("Lq"),
            "stable": result.get("stable"),
        })
    return pd.DataFrame(rows)


def sensitivity_mu(lam, mu, pct_changes=None, servers=1):
    if pct_changes is None:
        pct_changes = [-0.20, -0.10, 0.0, 0.10, 0.20]

    rows = []
    for change in pct_changes:
        new_mu = mu * (1 + change)
        result = mm1(lam, new_mu) if servers == 1 else mmc(lam, new_mu, servers)
        rows.append({
            "change_pct": change * 100,
            "mu": new_mu,
            "rho": result.get("rho"),
            "Wq_hours": result.get("Wq"),
            "Lq": result.get("Lq"),
            "stable": result.get("stable"),
        })
    return pd.DataFrame(rows)


def server_sensitivity(lam, mu, max_servers=8):
    rows = []
    for c in range(1, max_servers + 1):
        result = mmc(lam, mu, c)
        rows.append({
            "servers": c,
            "rho": result.get("rho"),
            "Wq_hours": result.get("Wq"),
            "Wq_minutes": result.get("Wq", float("nan")) * 60
                if result.get("Wq") is not None else None,
            "Lq": result.get("Lq"),
            "stable": result.get("stable"),
        })
    return pd.DataFrame(rows)
