import pandas as pd
from .queue_models import mmc


def staffing_cost_optimization(
    lam,
    mu,
    max_servers=8,
    server_cost_per_hour=100.0,
    waiting_cost_per_customer_hour=50.0,
):
    rows = []

    for c in range(1, max_servers + 1):
        result = mmc(lam, mu, c)

        if not result.get("stable"):
            rows.append({
                "servers": c,
                "stable": False,
                "rho": result.get("rho"),
                "Lq": None,
                "server_cost": c * server_cost_per_hour,
                "waiting_cost": float("inf"),
                "total_cost": float("inf"),
            })
            continue

        server_cost = c * server_cost_per_hour
        waiting_cost = result["Lq"] * waiting_cost_per_customer_hour
        total_cost = server_cost + waiting_cost

        rows.append({
            "servers": c,
            "stable": True,
            "rho": result["rho"],
            "Lq": result["Lq"],
            "Wq_hours": result["Wq"],
            "server_cost": server_cost,
            "waiting_cost": waiting_cost,
            "total_cost": total_cost,
        })

    df = pd.DataFrame(rows)
    feasible = df[df["total_cost"].notna() & (df["total_cost"] != float("inf"))]
    recommendation = int(feasible.loc[feasible["total_cost"].idxmin(), "servers"]) if not feasible.empty else None

    return {
        "table": df,
        "recommended_servers": recommendation,
        "minimum_total_cost": (
            float(feasible["total_cost"].min()) if not feasible.empty else None
        ),
    }
