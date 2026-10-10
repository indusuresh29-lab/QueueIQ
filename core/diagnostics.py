import numpy as np
import pandas as pd
from scipy.stats import chi2


# ============================================================
# QueueIQ — Dataset Diagnostics & Column Detection
# ============================================================

# Deterministic aliases for common queueing / service datasets.
# These are intentionally broad but explicit — no NLP / AI parser.
COLUMN_ALIASES = {
    "arrival": [
        "arrival",
        "arrival_time",
        "arrival_timestamp",
        "arrival_datetime",
        "arrived_at",
        "arrival_date_time",
        "call_started",
        "call_start",
        "call_start_time",
        "call_received",
        "call_received_time",
        "check_in",
        "checkin",
        "check_in_time",
        "registration_time",
        "customer_arrival",
        "customer_arrival_time",
        "patient_arrival",
        "patient_arrival_time",
        "entry_time",
        "entry_timestamp",
        "request_time",
        "request_timestamp",
        "created_at",
        "created_time",
    ],

    "service_start": [
        "service_start",
        "service_start_time",
        "service_start_timestamp",
        "service_begins",
        "service_begin",
        "service_began",
        "call_answered",
        "call_answer",
        "call_answer_time",
        "answered_at",
        "answer_time",
        "agent_answer_time",
        "doctor_start",
        "doctor_start_time",
        "treatment_start",
        "treatment_start_time",
        "counter_start",
        "counter_start_time",
        "processing_start",
        "processing_start_time",
    ],

    "service_end": [
        "service_end",
        "service_end_time",
        "service_end_timestamp",
        "service_ended",
        "service_finished",
        "service_finish",
        "call_ended",
        "call_end",
        "call_end_time",
        "call_completed",
        "call_completion",
        "call_completion_time",
        "ended_at",
        "end_time",
        "completion_time",
        "completed_at",
        "doctor_end",
        "doctor_end_time",
        "treatment_end",
        "treatment_end_time",
        "counter_end",
        "counter_end_time",
        "processing_end",
        "processing_end_time",
        "departure_time",
        "departure_timestamp",
    ],

    "waiting_time": [
        "waiting_time",
        "wait_time",
        "wait_length",
        "waiting_length",
        "waiting_duration",
        "waiting_duration_seconds",
        "waiting_duration_minutes",
        "waiting_duration_hours",
        "wait_duration",
        "wait_duration_seconds",
        "wait_duration_minutes",
        "wait_duration_hours",
        "queue_wait",
        "queue_wait_time",
        "queue_waiting_time",
        "queue_wait_duration",
        "time_in_queue",
        "time_waiting",
        "time_waited",
        "waiting_minutes",
        "waiting_seconds",
        "wait_minutes",
        "wait_seconds",
    ],

    "service_time": [
        "service_time",
        "service_length",
        "service_duration",
        "service_duration_seconds",
        "service_duration_minutes",
        "service_duration_hours",
        "handle_time",
        "handle_time_seconds",
        "handle_time_minutes",
        "handle_time_hours",
        "handling_time",
        "handling_duration",
        "handling_duration_seconds",
        "handling_duration_minutes",
        "processing_time",
        "processing_duration",
        "processing_duration_seconds",
        "processing_duration_minutes",
        "talk_time",
        "talk_time_seconds",
        "talk_time_minutes",
        "treatment_time",
        "treatment_duration",
        "service_minutes",
        "service_seconds",
    ],

    "queue_length": [
        "queue_length",
        "queue_size",
        "queue_count",
        "queue_length_at_arrival",
        "queue_size_at_arrival",
        "number_in_queue",
        "customers_in_queue",
        "patients_in_queue",
        "people_in_queue",
        "waiting_count",
        "waiting_customers",
        "waiting_patients",
        "backlog",
        "queue",
    ],

    "date": [
        "date",
        "business_date",
        "service_date",
        "transaction_date",
        "call_date",
        "arrival_date",
        "visit_date",
        "appointment_date",
        "day",
    ],

    "server_id": [
        "server_id",
        "server",
        "server_number",
        "server_no",
        "agent_id",
        "agent",
        "agent_number",
        "agent_no",
        "employee_id",
        "employee",
        "employee_number",
        "doctor_id",
        "doctor",
        "doctor_number",
        "counter_id",
        "counter",
        "counter_number",
        "desk_id",
        "desk",
        "staff_id",
        "staff",
        "operator_id",
        "operator",
    ],

    "sla": [
        "sla",
        "sla_met",
        "sla_status",
        "sla_compliance",
        "meets_standard",
        "met_sla",
        "within_sla",
        "service_level_met",
        "service_level",
        "target_met",
        "target_status",
    ],
}


# ------------------------------------------------------------
# Normalisation
# ------------------------------------------------------------

def _normalise_column_name(name):
    """
    Convert a column name into a deterministic comparison format.

    Example:
        'Waiting_Duration_Minutes'
        -> 'waiting duration minutes'
    """
    value = str(name).strip().lower()

    for char in ["_", "-", "/", "\\", ".", "(", ")", "[", "]"]:
        value = value.replace(char, " ")

    value = " ".join(value.split())

    return value


def _compact_column_name(name):
    """
    Remove separators entirely for stronger comparison.

    Example:
        'Waiting_Duration_Minutes'
        -> 'waitingdurationminutes'
    """
    return "".join(
        character
        for character in _normalise_column_name(name)
        if character.isalnum()
    )


# ------------------------------------------------------------
# Basic dataset profile
# ------------------------------------------------------------

def basic_profile(df):
    """
    Return a compact dataset profile.
    """
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_names": list(df.columns),
    }


# ------------------------------------------------------------
# Alias matching
# ------------------------------------------------------------

def _alias_match_score(column, alias):
    """
    Return a deterministic confidence score between 0 and 1.

    Matching hierarchy:
        1. Exact normalised match
        2. Exact compact match
        3. Alias contained in column name
        4. Column name contained in alias

    The scoring intentionally favours exact matches.
    """

    column_norm = _normalise_column_name(column)
    alias_norm = _normalise_column_name(alias)

    column_compact = _compact_column_name(column)
    alias_compact = _compact_column_name(alias)

    # Exact normalised match
    if column_norm == alias_norm:
        return 1.0

    # Exact compact match
    if column_compact == alias_compact:
        return 1.0

    # Strong token-level match
    column_tokens = set(column_norm.split())
    alias_tokens = set(alias_norm.split())

    if alias_tokens and alias_tokens.issubset(column_tokens):
        return 0.95

    # Alias contained in the full column name
    if alias_compact and alias_compact in column_compact:
        return 0.90

    # Column contained in alias
    if column_compact and column_compact in alias_compact:
        return 0.75

    return 0.0


# ------------------------------------------------------------
# Automatic column detection
# ------------------------------------------------------------

def detect_columns(df):
    """
    Detect likely queueing-related columns.

    Returns:
        {
            "detected": {...},
            "confidence": {...},
            "candidates": {...}
        }

    Detection is deterministic and editable later through the UI.
    """

    detected = {}
    confidence = {}
    candidates = {}

    for field, aliases in COLUMN_ALIASES.items():

        matches = []

        for column in df.columns:

            best_score = 0.0

            for alias in aliases:
                score = _alias_match_score(column, alias)
                best_score = max(best_score, score)

            if best_score > 0:
                matches.append(
                    {
                        "column": column,
                        "confidence": round(best_score, 4),
                    }
                )

        # Highest-confidence candidates first
        matches = sorted(
            matches,
            key=lambda item: item["confidence"],
            reverse=True,
        )

        candidates[field] = matches

        if matches:
            detected[field] = matches[0]["column"]
            confidence[field] = matches[0]["confidence"]
        else:
            detected[field] = None
            confidence[field] = 0.0

    return {
        "detected": detected,
        "confidence": confidence,
        "candidates": candidates,
    }


# ------------------------------------------------------------
# Mapping validation
# ------------------------------------------------------------

def validate_mapping(df, mapping):
    """
    Validate a user-selected column mapping.

    mapping example:
        {
            "arrival": "Arrival_Timestamp",
            "service_start": "Answer_Time",
            "service_end": "Call_End_Time",
            "waiting_time": "Waiting_Duration_Minutes",
            "service_time": "Handle_Time_Minutes",
            "server_id": "Employee_ID"
        }
    """

    errors = []
    warnings = []

    available_columns = set(df.columns)

    # Validate referenced columns
    for field, column in mapping.items():

        if column is None or column == "":
            continue

        if column not in available_columns:
            errors.append(
                f"{field}: selected column '{column}' does not exist in the dataset."
            )

    # Arrival is fundamental
    if not mapping.get("arrival"):
        errors.append(
            "Arrival column is required to derive the arrival rate λ."
        )

    # We need either timestamp information OR explicit durations
    has_service_timestamps = bool(
        mapping.get("service_start") and mapping.get("service_end")
    )

    has_service_duration = bool(mapping.get("service_time"))

    if not has_service_timestamps and not has_service_duration:
        warnings.append(
            "Service time cannot currently be derived. "
            "Map either Service Start + Service End or a Service Time column."
        )

    # Waiting can be derived from arrival + service start
    has_waiting_timestamp = bool(
        mapping.get("arrival") and mapping.get("service_start")
    )

    has_waiting_duration = bool(mapping.get("waiting_time"))

    if not has_waiting_timestamp and not has_waiting_duration:
        warnings.append(
            "Waiting time cannot currently be derived. "
            "Map either Arrival + Service Start or a Waiting Time column."
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# ------------------------------------------------------------
# Datetime conversion
# ------------------------------------------------------------

def _to_datetime(series):
    """
    Convert a pandas Series to datetime.

    Uses pandas' mixed-format parser when available.
    """

    try:
        return pd.to_datetime(
            series,
            errors="coerce",
            format="mixed",
        )
    except (TypeError, ValueError):
        return pd.to_datetime(
            series,
            errors="coerce",
        )


# ------------------------------------------------------------
# Numeric duration conversion
# ------------------------------------------------------------

def _duration_to_seconds(series, column_name):
    """
    Convert a numeric duration column into seconds.

    The unit is inferred from the column name.

    Examples:
        *_seconds -> seconds
        *_minutes -> minutes
        *_hours   -> hours

    If no unit is present, seconds are assumed.
    """

    numeric = pd.to_numeric(series, errors="coerce")

    name = _normalise_column_name(column_name)

    if "hour" in name or "hours" in name:
        return numeric * 3600.0

    if "minute" in name or "minutes" in name or "min" in name:
        return numeric * 60.0

    # Default assumption:
    # duration values without an explicit unit are seconds.
    return numeric


# ------------------------------------------------------------
# Derive queue metrics
# ------------------------------------------------------------

def derive_queue_metrics(df, mapping):
    """
    Derive standard queueing metrics from the selected mapping.

    Creates internal columns:

        _arrival_timestamp
        _service_start_timestamp
        _service_end_timestamp
        _waiting_seconds
        _service_seconds
        _interarrival_seconds
    """

    result = df.copy()

    arrival_col = mapping.get("arrival")
    service_start_col = mapping.get("service_start")
    service_end_col = mapping.get("service_end")
    waiting_col = mapping.get("waiting_time")
    service_col = mapping.get("service_time")

    # --------------------------------------------------------
    # Arrival timestamp
    # --------------------------------------------------------

    if arrival_col:
        result["_arrival_timestamp"] = _to_datetime(
            result[arrival_col]
        )
    else:
        result["_arrival_timestamp"] = pd.NaT

    # --------------------------------------------------------
    # Service start timestamp
    # --------------------------------------------------------

    if service_start_col:
        result["_service_start_timestamp"] = _to_datetime(
            result[service_start_col]
        )
    else:
        result["_service_start_timestamp"] = pd.NaT

    # --------------------------------------------------------
    # Service end timestamp
    # --------------------------------------------------------

    if service_end_col:
        result["_service_end_timestamp"] = _to_datetime(
            result[service_end_col]
        )
    else:
        result["_service_end_timestamp"] = pd.NaT

    # --------------------------------------------------------
    # Waiting time
    # --------------------------------------------------------

    if waiting_col:

        result["_waiting_seconds"] = _duration_to_seconds(
            result[waiting_col],
            waiting_col,
        )

    elif arrival_col and service_start_col:

        result["_waiting_seconds"] = (
            result["_service_start_timestamp"]
            - result["_arrival_timestamp"]
        ).dt.total_seconds()

    else:

        result["_waiting_seconds"] = np.nan

    # --------------------------------------------------------
    # Service time
    # --------------------------------------------------------

    if service_col:

        result["_service_seconds"] = _duration_to_seconds(
            result[service_col],
            service_col,
        )

    elif service_start_col and service_end_col:

        result["_service_seconds"] = (
            result["_service_end_timestamp"]
            - result["_service_start_timestamp"]
        ).dt.total_seconds()

    else:

        result["_service_seconds"] = np.nan

    # --------------------------------------------------------
    # Inter-arrival time
    # --------------------------------------------------------

    if arrival_col:

        valid_arrivals = (
            result["_arrival_timestamp"]
            .dropna()
            .sort_values()
        )

        interarrival = valid_arrivals.diff().dt.total_seconds()

        result["_interarrival_seconds"] = np.nan

        # Align by original index
        result.loc[
            interarrival.index,
            "_interarrival_seconds",
        ] = interarrival.values

    else:

        result["_interarrival_seconds"] = np.nan

    return result


# ------------------------------------------------------------
# Timestamp / queue diagnostics
# ------------------------------------------------------------

def timestamp_diagnostics(df, mapping=None):
    """
    Calculate arrival and service diagnostics.

    Supports:
        - timestamp-derived durations
        - explicit numeric waiting/service durations
    """

    if mapping is None:
        detection = detect_columns(df)
        mapping = detection["detected"]

    data = derive_queue_metrics(df, mapping)

    diagnostics = {
        "arrival_column": mapping.get("arrival"),
        "service_start_column": mapping.get("service_start"),
        "service_end_column": mapping.get("service_end"),
        "waiting_time_column": mapping.get("waiting_time"),
        "service_time_column": mapping.get("service_time"),
        "valid_arrivals": 0,
        "valid_waiting_times": 0,
        "valid_service_times": 0,
        "lambda_per_hour": None,
        "mu_per_hour": None,
        "mean_interarrival_seconds": None,
        "std_interarrival_seconds": None,
        "interarrival_cv": None,
        "mean_waiting_seconds": None,
        "mean_service_seconds": None,
        "std_service_seconds": None,
        "service_cv": None,
    }

    # --------------------------------------------------------
    # Arrival diagnostics
    # --------------------------------------------------------

    arrival_times = (
        data["_arrival_timestamp"]
        .dropna()
        .sort_values()
    )

    diagnostics["valid_arrivals"] = int(len(arrival_times))

    interarrival = (
        data["_interarrival_seconds"]
        .dropna()
    )

    interarrival = interarrival[
        interarrival > 0
    ]

    if len(interarrival) > 0:

        mean_interarrival = float(interarrival.mean())
        std_interarrival = float(interarrival.std(ddof=1))

        diagnostics["mean_interarrival_seconds"] = mean_interarrival
        diagnostics["std_interarrival_seconds"] = std_interarrival

        if mean_interarrival > 0:

            diagnostics["lambda_per_hour"] = (
                3600.0 / mean_interarrival
            )

        if mean_interarrival > 0:

            diagnostics["interarrival_cv"] = (
                std_interarrival / mean_interarrival
            )

    # --------------------------------------------------------
    # Waiting diagnostics
    # --------------------------------------------------------

    waiting = pd.to_numeric(
        data["_waiting_seconds"],
        errors="coerce",
    ).dropna()

    waiting = waiting[
        waiting >= 0
    ]

    diagnostics["valid_waiting_times"] = int(len(waiting))

    if len(waiting) > 0:
        diagnostics["mean_waiting_seconds"] = float(
            waiting.mean()
        )

    # --------------------------------------------------------
    # Service diagnostics
    # --------------------------------------------------------

    service = pd.to_numeric(
        data["_service_seconds"],
        errors="coerce",
    ).dropna()

    service = service[
        service > 0
    ]

    diagnostics["valid_service_times"] = int(len(service))

    if len(service) > 0:

        mean_service = float(service.mean())
        std_service = float(service.std(ddof=1))

        diagnostics["mean_service_seconds"] = mean_service
        diagnostics["std_service_seconds"] = std_service

        if mean_service > 0:

            diagnostics["mu_per_hour"] = (
                3600.0 / mean_service
            )

        if mean_service > 0:

            diagnostics["service_cv"] = (
                std_service / mean_service
            )

    return diagnostics


# ------------------------------------------------------------
# Poisson arrival diagnostic
# ------------------------------------------------------------

def poisson_chi_square_test(
    df,
    mapping=None,
    bins=8,
):
    """
    Test whether inter-arrival times are consistent with
    an exponential distribution.

    This is used as a diagnostic supporting a Poisson arrival
    assumption; it does NOT prove that arrivals are Poisson.
    """

    if mapping is None:
        detection = detect_columns(df)
        mapping = detection["detected"]

    data = derive_queue_metrics(df, mapping)

    interarrival = pd.to_numeric(
        data["_interarrival_seconds"],
        errors="coerce",
    ).dropna()

    interarrival = interarrival[
        interarrival > 0
    ]

    n = len(interarrival)

    result = {
        "sample_size": int(n),
        "chi_square": None,
        "degrees_of_freedom": None,
        "p_value": None,
        "alpha": 0.05,
        "decision": None,
        "interpretation": None,
    }

    if n < 40:

        result["decision"] = "Insufficient sample"
        result["interpretation"] = (
            "At least 40 valid inter-arrival observations "
            "are required for the current 8-bin diagnostic "
            "so that expected bin frequencies are at least 5."
        )

        return result

    mean_interarrival = float(
        interarrival.mean()
    )

    if mean_interarrival <= 0:

        result["decision"] = "Invalid data"
        result["interpretation"] = (
            "Mean inter-arrival time must be positive."
        )

        return result

    # --------------------------------------------------------
    # Equal-probability bins under exponential distribution
    # --------------------------------------------------------

    probabilities = np.linspace(
        0.0,
        0.95,
        bins,
    )[1:]

    edges = [
        0.0
    ]

    for probability in probabilities:

        edges.append(
            -mean_interarrival
            * np.log(1.0 - probability)
        )

    edges.append(np.inf)

    observed, _ = np.histogram(
        interarrival,
        bins=edges,
    )

    # Equal-probability bins
    expected = np.full(
        bins,
        n / bins,
        dtype=float,
    )

    # Chi-square statistic
    chi_square = float(
        np.sum(
            (observed - expected) ** 2
            / expected
        )
    )

    degrees_of_freedom = bins - 1 - 1

    p_value = float(
        chi2.sf(
            chi_square,
            degrees_of_freedom,
        )
    )

    result["chi_square"] = chi_square
    result["degrees_of_freedom"] = degrees_of_freedom
    result["p_value"] = p_value

    if p_value >= result["alpha"]:

        result["decision"] = (
            "Do not reject exponential inter-arrival distribution"
        )

        result["interpretation"] = (
            "The observed inter-arrival times are statistically "
            "consistent with an exponential distribution at the "
            "5% significance level. This supports, but does not "
            "prove, a Poisson arrival assumption."
        )

    else:

        result["decision"] = (
            "Reject exponential inter-arrival distribution"
        )

        result["interpretation"] = (
            "The observed inter-arrival times show statistically "
            "significant evidence against an exponential "
            "distribution at the 5% significance level. "
            "A Poisson arrival assumption should therefore "
            "be treated cautiously."
        )

    return result