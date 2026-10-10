import numpy as np
import pandas as pd

from core.diagnostics import (
    basic_profile,
    detect_columns,
    validate_mapping,
    timestamp_diagnostics,
    poisson_chi_square_test,
)


def test_basic_profile_counts_rows_columns_missing_and_duplicates():
    df = pd.DataFrame({
        "arrival_time": ["09:00", "09:01", "09:01"],
        "service_time": [30, 40, 40],
        "notes": [None, "B", "B"],
    })

    result = basic_profile(df)

    assert result["rows"] == 3
    assert result["columns"] == 3
    assert result["missing_cells"] == 1
    assert result["duplicate_rows"] == 1
    assert "arrival_time" in result["column_names"]


def test_detect_columns_recognises_queue_timestamps():
    df = pd.DataFrame({
        "arrival_time": ["09:00", "09:01"],
        "service_start_time": ["09:00:10", "09:01:10"],
        "service_end_time": ["09:00:40", "09:01:40"],
        "employee_id": [101, 102],
    })

    result = detect_columns(df)
    detected = result["detected"]

    assert detected["arrival"] == "arrival_time"
    assert detected["service_start"] == "service_start_time"
    assert detected["service_end"] == "service_end_time"
    assert result["confidence"]["arrival"] > 0


def test_validate_mapping_flags_missing_or_nonexistent_columns():
    df = pd.DataFrame({
        "arrival_time": ["09:00"],
        "service_time": [30],
    })

    result = validate_mapping(
        df,
        {
            "arrival": "missing_arrival",
            "service_time": "service_time",
        },
    )

    assert not result["valid"]
    assert any("does not exist" in error for error in result["errors"])


def test_timestamp_diagnostics_derives_rates_and_durations():
    arrivals = pd.date_range(
        "2026-01-01 09:00:00",
        periods=4,
        freq="min",
    )

    df = pd.DataFrame({
        "arrival_time": arrivals,
        "service_start_time": arrivals + pd.Timedelta(seconds=10),
        "service_end_time": arrivals + pd.Timedelta(seconds=40),
    })

    result = timestamp_diagnostics(
        df,
        {
            "arrival": "arrival_time",
            "service_start": "service_start_time",
            "service_end": "service_end_time",
        },
    )

    assert result["valid_arrivals"] == 4
    assert result["valid_waiting_times"] == 4
    assert result["valid_service_times"] == 4
    assert result["lambda_per_hour"] == 60
    assert result["mu_per_hour"] == 120
    assert result["mean_waiting_seconds"] == 10
    assert result["mean_service_seconds"] == 30


def test_poisson_diagnostic_reports_insufficient_sample():
    arrivals = pd.date_range(
        "2026-01-01 09:00:00",
        periods=20,
        freq="min",
    )
    df = pd.DataFrame({"arrival_time": arrivals})

    result = poisson_chi_square_test(
        df,
        mapping={"arrival": "arrival_time"},
    )

    assert result["sample_size"] == 19
    assert result["decision"] == "Insufficient sample"
    assert result["p_value"] is None


def test_poisson_diagnostic_runs_with_sufficient_sample():
    rng = np.random.default_rng(42)
    intervals = rng.exponential(scale=60, size=120)
    arrivals = pd.Timestamp("2026-01-01 09:00:00") + pd.to_timedelta(
        np.cumsum(intervals),
        unit="s",
    )
    df = pd.DataFrame({"arrival_time": arrivals})

    result = poisson_chi_square_test(
        df,
        mapping={"arrival": "arrival_time"},
    )

    assert result["sample_size"] == 119
    assert result["chi_square"] is not None
    assert result["degrees_of_freedom"] is not None
    assert 0 <= result["p_value"] <= 1
    assert result["decision"] is not None


def test_timestamp_diagnostics_auto_detects_columns():
    df = pd.DataFrame({
        "arrival_time": pd.date_range("2026-01-01 09:00:00", periods=5, freq="min"),
        "service_start_time": pd.date_range("2026-01-01 09:00:10", periods=5, freq="min"),
        "service_end_time": pd.date_range("2026-01-01 09:00:40", periods=5, freq="min"),
    })
    result = timestamp_diagnostics(df)
    assert result["arrival_column"] == "arrival_time"
    assert result["valid_arrivals"] == 5
    assert result["valid_waiting_times"] == 5
    assert result["valid_service_times"] == 5


def test_poisson_diagnostic_auto_detects_arrival_column():
    rng = np.random.default_rng(123)
    intervals = rng.exponential(scale=60, size=120)
    timestamps = pd.Timestamp("2026-01-01") + pd.to_timedelta(np.cumsum(intervals), unit="s")
    df = pd.DataFrame({"arrival_time": timestamps})
    result = poisson_chi_square_test(df)
    assert result["sample_size"] == 119
    assert result["chi_square"] is not None
    assert result["p_value"] is not None
    assert result["decision"] is not None


def test_poisson_diagnostic_rejects_regular_interarrival_times():
    timestamps = pd.date_range("2026-01-01", periods=121, freq="min")
    df = pd.DataFrame({"arrival_time": timestamps})
    result = poisson_chi_square_test(df, mapping={"arrival": "arrival_time"})
    assert result["sample_size"] == 120
    assert result["p_value"] < result["alpha"]
    assert result["decision"] == "Reject exponential inter-arrival distribution"


def test_derive_queue_metrics_without_arrival_column():
    from core.diagnostics import derive_queue_metrics
    df = pd.DataFrame({"service_time": [30, 45]})
    result = derive_queue_metrics(df, {"service_time": "service_time"})
    assert result["_arrival_timestamp"].isna().all()
    assert result["_interarrival_seconds"].isna().all()
    assert result["_service_seconds"].tolist() == [30, 45]
    assert result["_waiting_seconds"].isna().all()


def test_derive_queue_metrics_derives_durations_from_timestamps():
    from core.diagnostics import derive_queue_metrics
    df = pd.DataFrame({
        "arrival_time": ["2026-01-01 09:00:00", "2026-01-01 09:01:00"],
        "service_start_time": ["2026-01-01 09:00:10", "2026-01-01 09:01:15"],
        "service_end_time": ["2026-01-01 09:00:40", "2026-01-01 09:01:55"],
    })
    mapping = {
        "arrival": "arrival_time",
        "service_start": "service_start_time",
        "service_end": "service_end_time",
    }
    result = derive_queue_metrics(df, mapping)
    assert result["_waiting_seconds"].tolist() == [10.0, 15.0]
    assert result["_service_seconds"].tolist() == [30.0, 40.0]
    assert result["_interarrival_seconds"].iloc[1] == 60.0
