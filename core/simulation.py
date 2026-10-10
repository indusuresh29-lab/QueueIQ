import heapq
import numpy as np
import pandas as pd


def _service_time(mu, distribution, rng):
    if distribution == "Exponential":
        return rng.exponential(1 / mu)
    if distribution == "Deterministic":
        return 1 / mu
    if distribution == "Gamma":
        shape = 2.0
        return rng.gamma(shape, 1 / (mu * shape))
    if distribution == "Lognormal":
        sigma = 0.5
        mean_service = 1 / mu
        mean_log = np.log(mean_service) - 0.5 * sigma**2
        return rng.lognormal(mean_log, sigma)
    raise ValueError(f"Unsupported service distribution: {distribution}")


def simulate_queue(
    arrival_rate,
    service_rate,
    servers=1,
    n_customers=10000,
    service_distribution="Exponential",
    seed=42,
):
    if arrival_rate <= 0 or service_rate <= 0:
        raise ValueError("Rates must be positive.")
    if servers < 1:
        raise ValueError("Servers must be >= 1.")

    rng = np.random.default_rng(seed)
    interarrivals = rng.exponential(1 / arrival_rate, n_customers)
    arrivals = np.cumsum(interarrivals)

    server_free = np.zeros(servers)
    waits = np.zeros(n_customers)
    systems = np.zeros(n_customers)
    service_times = np.zeros(n_customers)

    for i, arrival in enumerate(arrivals):
        s = int(np.argmin(server_free))
        start = max(arrival, server_free[s])
        wait = start - arrival
        service = _service_time(service_rate, service_distribution, rng)
        finish = start + service

        waits[i] = wait
        service_times[i] = service
        systems[i] = finish - arrival
        server_free[s] = finish

    horizon = max(arrivals[-1], server_free.max())
    busy_time = service_times.sum()
    utilisation = busy_time / (servers * horizon)

    return {
        "n_customers": n_customers,
        "Wq_hours": float(waits.mean()),
        "W_hours": float(systems.mean()),
        "Wq_minutes": float(waits.mean() * 60),
        "W_minutes": float(systems.mean() * 60),
        "utilization": float(utilisation),
        "Lq": float(arrival_rate * waits.mean()),
        "L": float(arrival_rate * systems.mean()),
    }


def simulate_priority_queue(
    classes,
    servers=1,
    n_customers_per_class=3000,
    seed=42,
    preemptive=False,
):
    """
    Event-driven priority simulation.
    This MVP supports non-preemptive priority directly.
    Preemptive mode is intentionally rejected until a validated
    preemptive-resume engine is added.
    """
    if preemptive:
        raise NotImplementedError(
            "Preemptive-resume priority simulation is reserved for the validated advanced module."
        )

    rng = np.random.default_rng(seed)
    events = []

    for class_idx, item in enumerate(sorted(classes, key=lambda x: x["priority"])):
        lam = item["lambda"]
        if lam <= 0 or item["mu"] <= 0:
            raise ValueError("Priority class rates must be positive.")
        arrivals = np.cumsum(rng.exponential(1 / lam, n_customers_per_class))
        for seq, arrival in enumerate(arrivals):
            heapq.heappush(
                events,
                (float(arrival), int(item["priority"]), class_idx, seq, item)
            )

    server_free = [0.0] * servers
    waiting = []
    rows = []
    event_list = []

    while events or waiting:
        next_arrival = events[0][0] if events else float("inf")
        free_server = min(range(servers), key=lambda j: server_free[j])
        next_free = server_free[free_server]

        if events and next_arrival <= next_free:
            arrival, priority, class_idx, seq, item = heapq.heappop(events)
            heapq.heappush(
                waiting,
                (priority, arrival, class_idx, seq, item)
            )
            continue

        if waiting:
            priority, arrival, class_idx, seq, item = heapq.heappop(waiting)
            start = max(next_free, arrival)
            service = _service_time(item["mu"], item.get("distribution", "Exponential"), rng)
            finish = start + service
            server_free[free_server] = finish

            rows.append({
                "class": item["name"],
                "priority": item["priority"],
                "arrival": arrival,
                "start": start,
                "finish": finish,
                "wait_hours": start - arrival,
                "system_hours": finish - arrival,
            })
        else:
            # No waiting jobs; jump server clock to next arrival.
            if events:
                server_free[free_server] = next_arrival
            else:
                break

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    summary = (
        df.groupby(["class", "priority"], as_index=False)
        .agg(
            customers=("class", "size"),
            Wq_hours=("wait_hours", "mean"),
            W_hours=("system_hours", "mean"),
        )
    )
    summary["Wq_minutes"] = summary["Wq_hours"] * 60
    summary["W_minutes"] = summary["W_hours"] * 60
    return summary
