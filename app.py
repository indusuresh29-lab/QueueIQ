import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.queue_models import mm1, mmc, md1, finite_queue, finite_population
from core.diagnostics import (
    basic_profile,
    validate_mapping,
    timestamp_diagnostics,
    poisson_chi_square_test,
)
from core.simulation import simulate_queue, simulate_priority_queue
from core.sensitivity import sensitivity_lambda, sensitivity_mu, server_sensitivity
from core.optimization import staffing_cost_optimization
from core.validation import compare_theory_simulation, validate_little_law
from core.priority_queue import nonpreemptive_priority_m_m_1

st.set_page_config(
    page_title="QueueIQ",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("QueueIQ")
st.caption("Queueing Decision Intelligence | MBA Operations & Supply Chain Analytics")

with st.sidebar:
    st.header("Input")

    # ========================================================
    # INPUT MODE
    # ========================================================

    input_mode = st.radio(
        "Input mode",
        ["Manual Parameters", "Upload Dataset"],
        index=0,
    )

    st.divider()

    # ========================================================
    # QUEUE CONFIGURATION
    # ========================================================

    st.header("Queue Configuration")


    # ========================================================
    # MANUAL PARAMETER MODE
    # ========================================================

    if input_mode == "Manual Parameters":

        st.subheader("Queue Parameters")

        lam = st.number_input(
            "Arrival rate λ (customers/hour)",
            min_value=0.0001,
            value=12.0,
            step=1.0,
        )

        mu = st.number_input(
            "Service rate μ per server (customers/hour)",
            min_value=0.0001,
            value=15.0,
            step=1.0,
        )

        model = st.selectbox(
            "Queueing model",
            [
                "M/M/1",
                "M/M/c",
                "M/D/1",
                "Finite Capacity",
                "Finite Population",
            ],
        )

        if model == "M/M/c":
            servers = st.number_input(
                "Number of servers c",
                min_value=1,
                max_value=50,
                value=2,
                step=1,
            )
        else:
            servers = 1

        if model == "Finite Capacity":
            capacity = st.number_input(
                "Maximum system capacity K",
                min_value=1,
                max_value=500,
                value=10,
                step=1,
            )
        else:
            capacity = 10

        if model == "Finite Population":
            population = st.number_input(
                "Finite population N",
                min_value=1,
                max_value=500,
                value=20,
                step=1,
            )
        else:
            population = 20

        uploaded_df = None
        mapping = None
        data_diagnostics = None

    # ========================================================
    # DATASET MODE — QUEUEIQ STANDARD CSV
    # ========================================================

    else:

        st.subheader("QueueIQ Standard CSV")

        st.caption(
            "Upload a CSV using the QueueIQ standard data contract. "
            "QueueIQ derives waiting and service times from timestamps."
        )

        # --------------------------------------------------------
        # Standard format specification
        # --------------------------------------------------------

        st.markdown(
            """
            **Required columns**

            `customer_id` · `arrival_time` · `service_start` · `service_end`

            **Optional columns**

            `server_id` · `queue_length` · `sla_met` ·
            `priority_class` · `priority_level`
            """
        )

        from io import BytesIO
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter

        template_csv = (
            "customer_id,arrival_time,service_start,service_end,"
            "server_id,queue_length,sla_met,priority_class,priority_level\n"
            "C001,2026-10-08 09:00:15,2026-10-08 09:02:10,"
            "2026-10-08 09:07:30,S01,3,True,Normal,2\n"
            "C002,2026-10-08 09:01:20,2026-10-08 09:03:05,"
            "2026-10-08 09:08:10,S02,2,True,High,1\n"
            "C003,2026-10-08 09:02:45,2026-10-08 09:05:20,"
            "2026-10-08 09:11:00,S01,4,False,Normal,2\n"
        )

        st.download_button(
            "Download QueueIQ CSV Template",
            data=template_csv,
            file_name="QueueIQ_standard_template.csv",
            mime="text/csv",
            width="stretch",
        )

        # Build an Excel workbook with data and instructions on one worksheet.
        workbook = Workbook()
        ws = workbook.active
        ws.title = "QueueIQ Template"

        headers = [
            "customer_id",
            "arrival_time",
            "service_start",
            "service_end",
            "server_id",
            "queue_length",
            "sla_met",
            "priority_class",
            "priority_level",
        ]

        sample_rows = [
            ["C001", "2026-10-08 09:00:15", "2026-10-08 09:02:10",
             "2026-10-08 09:07:30", "S01", 3, True, "Normal", 2],
            ["C002", "2026-10-08 09:01:20", "2026-10-08 09:03:05",
             "2026-10-08 09:08:10", "S02", 2, True, "High", 1],
            ["C003", "2026-10-08 09:02:45", "2026-10-08 09:05:20",
             "2026-10-08 09:11:00", "S01", 4, False, "Normal", 2],
        ]

        ws.append(headers)
        for row in sample_rows:
            ws.append(row)

        # Keep instructions outside the nine-column upload data range.
        ws.merge_cells("K1:N1")
        ws["K1"] = "QUEUEIQ TEMPLATE — INSTRUCTIONS"
        ws["K1"].font = Font(bold=True, color="FFFFFF", size=14)
        ws["K1"].fill = PatternFill("solid", fgColor="17365D")
        ws["K1"].alignment = Alignment(vertical="center")
        ws.row_dimensions[1].height = 28

        instruction_headers = ["Column / Topic", "Meaning", "How to fill it", "Scenario notes"]
        for col, value in enumerate(instruction_headers, start=11):
            cell = ws.cell(row=3, column=col, value=value)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="2878A5")
            cell.alignment = Alignment(wrap_text=True, vertical="top")

        instructions = [
            ["REQUIRED: customer_id", "Unique customer, patient, order or request ID",
             "Use one unique ID per row.", "One customer or service request per row."],
            ["REQUIRED: arrival_time", "Time the customer joins the queue",
             "Use YYYY-MM-DD HH:MM:SS.", "Record actual arrival times."],
            ["REQUIRED: service_start", "Time service begins",
             "Must be at or after arrival_time.", "Do not estimate missing timestamps."],
            ["REQUIRED: service_end", "Time service finishes",
             "Must be after service_start.", "Use the same timezone throughout."],
            ["OPTIONAL: server_id", "Counter, agent, employee or machine ID",
             "Use consistent IDs such as S01, S02.", "Use one ID for a single-server system; distinct IDs for multiple servers."],
            ["OPTIONAL: queue_length", "Observed number waiting in the queue",
             "Use one consistent counting definition.", "Record it at a consistent observation point."],
            ["OPTIONAL: sla_met", "Whether the service-level target was met",
             "Use True or False consistently.", "Define the SLA target used."],
            ["OPTIONAL: priority_class", "Category such as Emergency, VIP, Standard or Normal",
             "Use consistent labels.", "Populate for priority-related analysis."],
            ["OPTIONAL: priority_level", "Numeric priority rank",
             "Use 1 for highest priority, 2 for the next, and so on.", "Keep each class mapped to the same rank."],
            ["OPTIONAL: waiting_time", "Time from arrival to service start",
             "If supplied, use seconds.", "Should agree with service_start minus arrival_time."],
            ["OPTIONAL: service_time", "Time from service start to service end",
             "If supplied, use seconds.", "Should agree with service_end minus service_start."],
            ["SINGLE SERVER / FIFO", "One server serving customers in arrival order",
             "Use one consistent server_id.", "Confirm service order matches the intended FIFO scenario."],
            ["MULTIPLE SERVERS", "Two or more servers handling customers",
             "Use distinct IDs such as S01, S02 and S03.", "Overlapping service times across servers are valid."],
            ["PRIORITY QUEUE", "Customers may be served according to priority",
             "Populate priority_class, priority_level, or both.", "Labels alone do not prove the system actually uses priority scheduling."],
            ["PEAK / OFF-PEAK", "Different demand periods",
             "Include observations from the periods being compared.", "Preserve real timestamps and arrival patterns."],
            ["DATA QUALITY", "Checks before uploading",
             "Keep the nine column names unchanged; do not leave required timestamps blank.", "Do not invent or evenly space records to make data look cleaner."],
            ["MODEL INTERPRETATION", "Results depend on the data and model assumptions",
             "Provide enough representative observations.", "An uploaded dataset does not automatically establish that a queueing model fits."],
            ["CSV EXPORT", "Preparing data for QueueIQ upload",
             "Export only columns A–I, including the header row.", "Do not include the instructions in the uploaded CSV."],
        ]

        for row_num, row_data in enumerate(instructions, start=4):
            for col_num, value in enumerate(row_data, start=11):
                cell = ws.cell(row=row_num, column=col_num, value=value)
                cell.alignment = Alignment(
                    wrap_text=True,
                    vertical="top",
                )
                if row_num % 2 == 0:
                    cell.fill = PatternFill("solid", fgColor="EAF2F8")

        # Visual formatting for the data table.
        for cell in ws[1][:9]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="17365D")
            cell.alignment = Alignment(wrap_text=True, vertical="top")

        data_widths = [16, 22, 22, 22, 13, 15, 12, 18, 16]
        for index, width in enumerate(data_widths, start=1):
            ws.column_dimensions[get_column_letter(index)].width = width

        ws.column_dimensions["J"].width = 3
        ws.column_dimensions["K"].width = 24
        ws.column_dimensions["L"].width = 30
        ws.column_dimensions["M"].width = 38
        ws.column_dimensions["N"].width = 42

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = "A1:I4"

        for row_num in range(4, 4 + len(instructions)):
            ws.row_dimensions[row_num].height = 48

        ws.sheet_view.showGridLines = True

        excel_buffer = BytesIO()
        workbook.save(excel_buffer)
        excel_buffer.seek(0)

        st.download_button(
            "Download QueueIQ Excel Template with Instructions",
            data=excel_buffer.getvalue(),
            file_name="QueueIQ_template_with_instructions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )

        st.divider()

        uploaded_file = st.file_uploader(
            "Upload QueueIQ Standard CSV",
            type=["csv"],
            help=(
                "The CSV must contain customer_id, arrival_time, "
                "service_start and service_end."
            ),
        )

        uploaded_df = None
        mapping = None
        data_diagnostics = None
        profile = None

        if uploaded_file is not None:

            try:

                uploaded_df = pd.read_csv(
                    uploaded_file
                )

                st.success(
                    f"Loaded {len(uploaded_df):,} rows × "
                    f"{len(uploaded_df.columns):,} columns"
                )

                # ------------------------------------------------
                # Required schema
                # ------------------------------------------------

                required_columns = [
                    "customer_id",
                    "arrival_time",
                    "service_start",
                    "service_end",
                ]

                optional_columns = [
                    "server_id",
                    "queue_length",
                    "sla_met",
                    "priority_class",
                    "priority_level",
                ]

                missing_required = [
                    column
                    for column in required_columns
                    if column not in uploaded_df.columns
                ]

                unexpected_columns = [
                    column
                    for column in uploaded_df.columns
                    if column not in required_columns
                    and column not in optional_columns
                ]

                if missing_required:

                    st.error(
                        "Invalid QueueIQ CSV. Missing required "
                        "columns: "
                        + ", ".join(missing_required)
                    )

                    st.info(
                        "Download the QueueIQ CSV template above "
                        "and use the exact column names."
                    )

                    mapping = None

                else:

                    # ------------------------------------------------
                    # Fixed deterministic mapping
                    # ------------------------------------------------

                    mapping = {
                        "arrival": "arrival_time",
                        "service_start": "service_start",
                        "service_end": "service_end",
                        "waiting_time": None,
                        "service_time": None,
                        "queue_length": (
                            "queue_length"
                            if "queue_length"
                            in uploaded_df.columns
                            else None
                        ),
                        "date": None,
                        "server_id": (
                            "server_id"
                            if "server_id"
                            in uploaded_df.columns
                            else None
                        ),
                        "sla": (
                            "sla_met"
                            if "sla_met"
                            in uploaded_df.columns
                            else None
                        ),
                    }

                    # ------------------------------------------------
                    # Dataset profile
                    # ------------------------------------------------

                    profile = basic_profile(
                        uploaded_df
                    )

                    with st.expander(
                        "Dataset Profile",
                        expanded=False,
                    ):

                        p1, p2, p3, p4 = st.columns(4)

                        p1.metric(
                            "Rows",
                            f"{profile['rows']:,}",
                        )

                        p2.metric(
                            "Columns",
                            f"{profile['columns']:,}",
                        )

                        p3.metric(
                            "Missing Cells",
                            f"{profile['missing_cells']:,}",
                        )

                        p4.metric(
                            "Duplicate Rows",
                            f"{profile['duplicate_rows']:,}",
                        )

                    # ------------------------------------------------
                    # Schema validation
                    # ------------------------------------------------

                    mapping_check = validate_mapping(
                        uploaded_df,
                        mapping,
                    )

                    if mapping_check["errors"]:

                        for error in mapping_check["errors"]:
                            st.error(error)

                    if mapping_check["warnings"]:

                        for warning in mapping_check["warnings"]:
                            st.warning(warning)

                    # ------------------------------------------------
                    # Timestamp validation
                    # ------------------------------------------------

                    if mapping_check["valid"]:

                        arrival_values = pd.to_datetime(
                            uploaded_df["arrival_time"],
                            errors="coerce",
                        )

                        start_values = pd.to_datetime(
                            uploaded_df["service_start"],
                            errors="coerce",
                        )

                        end_values = pd.to_datetime(
                            uploaded_df["service_end"],
                            errors="coerce",
                        )

                        invalid_timestamps = (
                            arrival_values.isna()
                            | start_values.isna()
                            | end_values.isna()
                        )

                        invalid_sequence = (
                            (start_values < arrival_values)
                            | (end_values < start_values)
                        )

                        invalid_count = int(
                            (
                                invalid_timestamps
                                | invalid_sequence
                            ).sum()
                        )

                        if invalid_count > 0:

                            st.error(
                                f"{invalid_count:,} rows contain "
                                "invalid or chronologically inconsistent "
                                "timestamps."
                            )

                            mapping_check["valid"] = False

                        else:

                            st.success(
                                "QueueIQ Standard CSV validated successfully."
                            )

                    # ------------------------------------------------
                    # Standard mapping display
                    # ------------------------------------------------

                    with st.expander(
                        "QueueIQ Data Contract",
                        expanded=False,
                    ):

                        schema_rows = []

                        for column in required_columns:

                            schema_rows.append(
                                {
                                    "Field": column,
                                    "Status": "Required",
                                    "Detected": "Yes",
                                }
                            )

                        for column in optional_columns:

                            schema_rows.append(
                                {
                                    "Field": column,
                                    "Status": "Optional",
                                    "Detected": (
                                        "Yes"
                                        if column
                                        in uploaded_df.columns
                                        else "No"
                                    ),
                                }
                            )

                        st.dataframe(
                            pd.DataFrame(schema_rows),
                            width="stretch",
                            hide_index=True,
                        )

                    # ------------------------------------------------
                    # Derive λ and μ
                    # ------------------------------------------------

                    if mapping_check["valid"]:

                        data_diagnostics = (
                            timestamp_diagnostics(
                                uploaded_df,
                                mapping,
                            )
                        )

                        estimated_lambda = (
                            data_diagnostics.get(
                                "lambda_per_hour"
                            )
                        )

                        estimated_mu = (
                            data_diagnostics.get(
                                "mu_per_hour"
                            )
                        )

                        if estimated_lambda is not None:
                            lam = float(estimated_lambda)
                        else:
                            lam = None

                        if estimated_mu is not None:
                            mu = float(estimated_mu)
                        else:
                            mu = None

                        st.subheader(
                            "Estimated Queue Parameters"
                        )

                        d1, d2 = st.columns(2)

                        d1.metric(
                            "Estimated λ",
                            (
                                f"{estimated_lambda:.3f}/hour"
                                if estimated_lambda is not None
                                else "Unavailable"
                            ),
                        )

                        d2.metric(
                            "Estimated μ",
                            (
                                f"{estimated_mu:.3f}/hour"
                                if estimated_mu is not None
                                else "Unavailable"
                            ),
                        )

                        # ------------------------------------------------
                        # Analytical model
                        # ------------------------------------------------

                        model = st.selectbox(
                            "Analytical queueing model",
                            [
                                "M/M/1",
                                "M/M/c",
                                "M/D/1",
                                "Finite Capacity",
                                "Finite Population",
                            ],
                        )

                        if model == "M/M/c":

                            servers = st.number_input(
                                "Number of servers c",
                                min_value=1,
                                max_value=50,
                                value=2,
                                step=1,
                            )

                        else:

                            servers = 1

                        capacity = (
                            st.number_input(
                                "Maximum system capacity K",
                                min_value=1,
                                max_value=500,
                                value=10,
                                step=1,
                            )
                            if model == "Finite Capacity"
                            else 10
                        )

                        population = (
                            st.number_input(
                                "Finite population N",
                                min_value=1,
                                max_value=500,
                                value=20,
                                step=1,
                            )
                            if model == "Finite Population"
                            else 20
                        )

                    else:

                        st.warning(
                            "The uploaded dataset could not provide valid "
                            "arrival and service estimates. Please check "
                            "the QueueIQ Standard CSV format."
                        )

                        lam = None
                        mu = None
                        model = "M/M/1"
                        servers = 1
                        capacity = 10
                        population = 20

            except Exception as exc:

                st.error(
                    f"Could not read the QueueIQ CSV: {exc}"
                )

                lam = None
                mu = None
                model = "M/M/1"
                servers = 1
                capacity = 10
                population = 20

        else:

            st.info(
                "Upload the QueueIQ Standard CSV to begin "
                "data-driven queue analysis."
            )

            lam = None
            mu = None
            model = "M/M/1"
            servers = 1
            capacity = 10
            population = 20

    # ========================================================
    # QUEUE CONFIGURATION
    # ========================================================

    st.header("Queue Configuration")

    has_priority_data = False

    if input_mode == "Upload Dataset" and uploaded_df is not None:
        priority_columns = [
            column
            for column in ["priority_class", "priority_level"]
            if column in uploaded_df.columns
            and uploaded_df[column].notna().any()
            and uploaded_df[column].astype(str).str.strip().ne("").any()
        ]
        has_priority_data = bool(priority_columns)

    queue_options = ["FIFO / FCFS", "Priority"]

    queue_mode = st.selectbox(
        "Queue discipline",
        queue_options,
        index=1 if has_priority_data else 0,
        key="queue_mode_selector",
    )

    if has_priority_data:
        st.info(
            "Priority information detected in the uploaded dataset. "
            "Priority is selected by default; you can change the "
            "queue discipline above."
        )

    # ========================================================
    # PRIORITY QUEUE
    # ========================================================

    if queue_mode == "Priority":

        st.divider()
        st.subheader("Priority Queue Configuration")

        st.caption(
            "Configure customer classes from highest to lowest priority. "
            "QueueIQ currently uses a non-preemptive priority model."
        )

        priority_rows = st.number_input(
            "Number of priority classes",
            min_value=2,
            max_value=6,
            value=2,
            step=1,
        )

        priority_classes = []

        for i in range(int(priority_rows)):

            pc1, pc2, pc3, pc4 = st.columns(4)

            with pc1:
                class_name = st.text_input(
                    f"Class {i + 1}",
                    value=(
                        "High"
                        if i == 0
                        else "Normal"
                        if i == 1
                        else f"Class {i + 1}"
                    ),
                    key=f"priority_name_{i}",
                )

            with pc2:
                class_lambda = st.number_input(
                    "λ / hour",
                    min_value=0.0001,
                    value=3.0 if i == 0 else 5.0,
                    step=1.0,
                    key=f"priority_lambda_{i}",
                )

            with pc3:
                class_mu = st.number_input(
                    "μ / hour",
                    min_value=0.0001,
                    value=15.0,
                    step=1.0,
                    key=f"priority_mu_{i}",
                )

            with pc4:
                class_distribution = st.selectbox(
                    "Service distribution",
                    [
                        "Exponential",
                        "Deterministic",
                        "Gamma",
                        "Lognormal",
                    ],
                    key=f"priority_distribution_{i}",
                )

            priority_classes.append(
                {
                    "name": class_name,
                    "priority": i + 1,
                    "lambda": float(class_lambda),
                    "mu": float(class_mu),
                    "service_distribution": class_distribution,
                }
            )

    # ========================================================
    # STAFFING / SIMULATION SETTINGS
    # ========================================================

    st.divider()
    st.header("Decision Settings")

    server_cost = st.number_input(
        "Server cost per hour",
        min_value=0.0,
        value=500.0,
        step=50.0,
    )

    waiting_cost = st.number_input(
        "Waiting cost per customer-hour",
        min_value=0.0,
        value=300.0,
        step=25.0,
    )

    max_servers = st.slider(
        "Compare servers from 1 to",
        min_value=1,
        max_value=12,
        value=6,
    )

    st.subheader("Simulation")

    run_sim = st.checkbox(
        "Run discrete-event simulation",
        value=True,
    )

    sim_customers = st.slider(
        "Simulation customers",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000,
    )

    service_dist = st.selectbox(
        "Simulation service distribution",
        [
            "Exponential",
            "Deterministic",
            "Gamma",
            "Lognormal",
        ],
    )

# ============================================================
# MAIN APPLICATION WORKSPACE
# ============================================================

st.title("QueueIQ — Queueing Decision Intelligence")
st.caption(
    "Analytical queueing models, simulation, sensitivity analysis, "
    "staffing optimisation, diagnostics and governance."
)


# ============================================================
# ANALYTICAL ENGINE
# ============================================================

# The sidebar has already resolved λ and μ from either
# manual parameters or the uploaded dataset.
# Do not overwrite or re-cast these values here.

# ============================================================
# INPUT / QUEUE STATE
# ============================================================

inputs_available = (
    lam is not None
    and mu is not None
    and float(lam) > 0
    and float(mu) > 0
)

analysis_result = None
priority_result = None

try:

    if queue_mode == "Priority":
        priority_result = nonpreemptive_priority_m_m_1(
            priority_classes
        )
        analysis_result = priority_result

    elif not inputs_available:
        analysis_result = None

    elif model == "M/M/1":
        analysis_result = mm1(lam, mu)

    elif model == "M/M/c":
        analysis_result = mmc(
            lam,
            mu,
            int(servers),
        )

    elif model == "M/D/1":
        analysis_result = md1(lam, mu)

    elif model == "Finite Capacity":
        analysis_result = finite_queue(
            lam,
            mu,
            int(capacity),
            int(servers),
        )

    elif model == "Finite Population":
        analysis_result = finite_population(
            lam,
            mu,
            int(population),
        )

except Exception as exc:
    analysis_result = None
    st.error(
        f"Analytical model could not be evaluated: {exc}"
    )


# ============================================================
# DECISION DASHBOARD
# ============================================================

st.divider()

st.header("Decision Dashboard")

def safe_metric(value, fmt=".2f", suffix=""):
    if value is None:
        return "Not finite"
    try:
        value = float(value)
        if not np.isfinite(value):
            return "Not finite"
        return format(value, fmt) + suffix
    except (TypeError, ValueError):
        return "Not available"

if analysis_result is None:
    st.warning(
        "Enter valid queue parameters or upload a dataset with "
        "sufficient arrival and service information."
    )

elif queue_mode == "Priority":
    st.subheader("Priority Queue Performance")

    if not priority_result or not priority_result.get("stable", False):
        st.error(
            "UNSTABLE QUEUE: aggregate utilisation is at least 100%. "
            "Finite steady-state performance metrics are unavailable."
        )
    else:
        st.dataframe(
            pd.DataFrame(priority_result.get("classes", [])),
            width="stretch",
            hide_index=True,
        )

else:
    rho_value = analysis_result.get("rho")
    lq_value = analysis_result.get("Lq")
    l_value = analysis_result.get("L")
    wq_value = analysis_result.get("Wq")
    w_value = analysis_result.get("W")

    if (
        not analysis_result.get("stable", True)
        or (rho_value is not None and rho_value >= 1)
    ):
        status = "UNSTABLE"
    elif rho_value is not None and rho_value >= 0.95:
        status = "CRITICAL"
    elif rho_value is not None and rho_value >= 0.85:
        status = "HIGH RISK"
    elif rho_value is not None and rho_value >= 0.70:
        status = "WATCH"
    else:
        status = "STABLE"

    st.subheader("Queue Health")
    col1, col2 = st.columns(2)
    col1.metric("System Status", status)
    col2.metric("Utilisation (ρ)", safe_metric(rho_value, ".1%"))

    if status == "UNSTABLE":
        st.error(
            "Arrival demand meets or exceeds effective service capacity. "
            "A finite steady-state queue does not exist for this model "
            "under the current parameters."
        )

        if lam is not None and mu is not None:
            server_count = (
                int(servers)
                if model in ("M/M/c", "Finite Capacity")
                else 1
            )
            capacity_rate = float(mu) * server_count
            capacity_gap = float(lam) - capacity_rate

            col1, col2, col3 = st.columns(3)
            col1.metric("Arrival Rate λ", f"{float(lam):,.2f}/hour")
            col2.metric("Service Capacity", f"{capacity_rate:,.2f}/hour")
            col3.metric("Capacity Gap", f"{capacity_gap:,.2f}/hour")

        st.write(
            "**Recommended action:** check the input units, increase "
            "effective service capacity, or reduce/redistribute demand. "
            "Recalculate before making staffing decisions."
        )

    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Queue Length (Lq)", safe_metric(lq_value))
        col2.metric(
            "Average Wait (Wq)",
            safe_metric(
                None if wq_value is None else wq_value * 60,
                ".2f",
                " min",
            ),
        )
        col3.metric(
            "Time in System (W)",
            safe_metric(
                None if w_value is None else w_value * 60,
                ".2f",
                " min",
            ),
        )

        if status == "CRITICAL":
            st.error(
                "CRITICAL: utilisation is at least 95%. "
                "Small demand increases may sharply increase waiting time."
            )
        elif status == "HIGH RISK":
            st.warning(
                "HIGH RISK: utilisation is elevated. Review peak demand "
                "and additional service capacity."
            )
        elif status == "WATCH":
            st.info("WATCH: monitor demand and waiting-time trends.")
        else:
            st.success("The queue is operating within a stable range.")

        with st.expander("Detailed analytical results"):
            details = pd.DataFrame({
                "Metric": [
                    "Arrival rate λ (customers/hour)",
                    "Service rate μ per server (customers/hour)",
                    "Queue length Lq",
                    "System population L",
                    "Waiting time Wq (minutes)",
                    "Time in system W (minutes)",
                ],
                "Value": [
                    safe_metric(lam),
                    safe_metric(mu),
                    safe_metric(lq_value, ".3f"),
                    safe_metric(l_value, ".3f"),
                    safe_metric(
                        None if wq_value is None else wq_value * 60
                    ),
                    safe_metric(
                        None if w_value is None else w_value * 60
                    ),
                ],
            })
            st.dataframe(details, width="stretch", hide_index=True)

        if wq_value is not None and np.isfinite(wq_value):
            st.info(
                f"Expected average waiting time before service: "
                f"{wq_value * 60:.2f} minutes."
            )

st.header("Simulation & Analytical Validation")

st.caption(
    "Compare theoretical queueing performance with discrete-event "
    "simulation under alternative service-time assumptions."
)

simulation_tabs = st.tabs(
    [
        "Analytical vs Simulation",
        "Service-Time Distributions",
        "Little's Law",
    ]
)

# ------------------------------------------------------------
# Analytical vs simulation
# ------------------------------------------------------------

with simulation_tabs[0]:

    st.subheader("Analytical vs Simulation")

    if queue_mode == "Priority":

        st.info(
            "Priority simulation is available separately. "
            "The analytical-vs-simulation benchmark below applies "
            "to the standard queueing models."
        )

    elif not inputs_available:
        st.warning(
            "Simulation requires valid arrival and service rates."
        )

    elif (
        model in ("M/M/1", "M/M/c", "M/D/1")
        and float(lam) >= float(mu) * (
            int(servers) if model == "M/M/c" else 1
        )
    ):
        st.error(
            "Steady-state simulation comparison skipped because "
            "the selected queue is unstable."
        )

    elif run_sim:

        try:

            if model == "M/M/1":
                theory = mm1(lam, mu)

            elif model == "M/M/c":
                theory = mmc(
                    lam,
                    mu,
                    int(servers),
                )

            elif model == "M/D/1":
                theory = md1(lam, mu)

            else:
                theory = None

            if theory is not None:

                simulation_result = simulate_queue(
                    arrival_rate=lam,
                    service_rate=mu,
                    servers=int(servers),
                    n_customers=int(sim_customers),
                    service_distribution=service_dist,
                    seed=42,
                )

                validation = compare_theory_simulation(
                    theory,
                    simulation_result,
                    tolerance=0.10,
                )

                validation_rows = []

                for metric, result in validation.items():

                    validation_rows.append(
                        {
                            "Metric": metric,
                            "Analytical": result["theoretical"],
                            "Simulation": result["simulated"],
                            "Relative Error": result["relative_error"],
                            "Within 10%": result["within_tolerance"],
                        }
                    )

                validation_df = pd.DataFrame(
                    validation_rows
                )

                display_df = validation_df.copy()

                display_df["Analytical"] = display_df[
                    "Analytical"
                ].map(
                    lambda x: f"{x:.4f}"
                )

                display_df["Simulation"] = display_df[
                    "Simulation"
                ].map(
                    lambda x: f"{x:.4f}"
                )

                display_df["Relative Error"] = display_df[
                    "Relative Error"
                ].map(
                    lambda x: f"{x:.2%}"
                )

                st.dataframe(
                    display_df,
                    width="stretch",
                    hide_index=True,
                )

                if validation_df["Within 10%"].all():

                    st.success(
                        "Simulation is within the 10% validation "
                        "tolerance for all comparable metrics."
                    )

                else:

                    st.warning(
                        "One or more simulation metrics exceed the "
                        "10% validation tolerance. Increase the "
                        "simulation sample size or review model assumptions."
                    )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Simulation Customers",
                    f"{int(sim_customers):,}",
                )

                c2.metric(
                    "Simulated Wq",
                    f"{simulation_result['Wq_minutes']:.2f} min",
                )

                c3.metric(
                    "Simulated Utilisation",
                    f"{simulation_result['utilization']:.1%}",
                )

            else:

                st.info(
                    "Analytical-vs-simulation comparison is currently "
                    "available for M/M/1, M/M/c and M/D/1."
                )

        except Exception as exc:

            st.error(
                f"Simulation could not be completed: {exc}"
            )

    else:

        st.info(
            "Enable 'Run discrete-event simulation' in the sidebar "
            "to run the simulation."
        )


# ------------------------------------------------------------
# Service-time distributions
# ------------------------------------------------------------

with simulation_tabs[1]:

    st.subheader("Service-Time Distribution Experiment")

    st.write(
        "The mean service rate is held constant while the shape of "
        "the service-time distribution changes."
    )

    distribution_rows = []

    for distribution in [
        "Exponential",
        "Deterministic",
        "Gamma",
        "Lognormal",
    ]:

        if not inputs_available:
            st.warning(
                "Distribution experiments require valid arrival and service rates."
            )
            break

        if (
            model in ("M/M/1", "M/M/c", "M/D/1")
            and float(lam) >= float(mu) * (
                int(servers) if model == "M/M/c" else 1
            )
        ):
            st.error(
                "Distribution experiments skipped because the selected "
                "queue is unstable. Increase capacity or reduce demand first."
            )
            break

        result = simulate_queue(
            arrival_rate=lam,
            service_rate=mu,
            servers=int(servers),
            n_customers=int(sim_customers),
            service_distribution=distribution,
            seed=42,
        )

        distribution_rows.append(
            {
                "Distribution": distribution,
                "Wq (min)": result["Wq_minutes"],
                "W (min)": result["W_minutes"],
                "Utilisation": result["utilization"],
                "Lq": result["Lq"],
                "L": result["L"],
            }
        )

    if distribution_rows:
        distribution_df = pd.DataFrame(
            distribution_rows
        )

        distribution_display = distribution_df.copy()

        distribution_display["Wq (min)"] = distribution_display[
            "Wq (min)"
        ].map(
            lambda x: f"{x:.2f}"
        )

        distribution_display["W (min)"] = distribution_display[
            "W (min)"
        ].map(
            lambda x: f"{x:.2f}"
        )

        distribution_display["Utilisation"] = distribution_display[
            "Utilisation"
        ].map(
            lambda x: f"{x:.1%}"
        )

        distribution_display["Lq"] = distribution_display[
            "Lq"
        ].map(
            lambda x: f"{x:.3f}"
        )

        distribution_display["L"] = distribution_display[
            "L"
        ].map(
            lambda x: f"{x:.3f}"
        )

        st.dataframe(
            distribution_display,
            width="stretch",
            hide_index=True,
        )

        st.line_chart(
            distribution_df.set_index(
                "Distribution"
            )[
                "Wq (min)"
            ],
            height=320,
        )

        st.info(
            "Queue performance is sensitive not only to average service "
            "capacity but also to service-time variability."
        )


    # ------------------------------------------------------------
    else:
        st.info(
            "No service-time distribution results are available. "
            "Provide valid, stable queue inputs to run this experiment."
        )

# Little's Law
# ------------------------------------------------------------

with simulation_tabs[2]:

    st.subheader("Little's Law Validation")

    if queue_mode == "Priority":

        st.info(
            "Little's Law validation is applied to the standard "
            "single-class analytical model."
        )

    else:

        try:

            if model == "M/M/1":
                theory = mm1(lam, mu)

            elif model == "M/M/c":
                theory = mmc(
                    lam,
                    mu,
                    int(servers),
                )

            elif model == "M/D/1":
                theory = md1(lam, mu)

            else:
                theory = None

            if theory is not None:

                little_law = validate_little_law(
                    lam,
                    theory["L"],
                    theory["W"],
                )

                l1, l2, l3 = st.columns(3)

                l1.metric(
                    "Observed L",
                    f"{little_law['observed_L']:.4f}",
                )

                l2.metric(
                    "λ × W",
                    f"{little_law['expected_L']:.4f}",
                )

                l3.metric(
                    "Relative Error",
                    f"{little_law['relative_error']:.4%}",
                )

                if little_law["valid"]:

                    st.success(
                        "Little's Law is satisfied within the "
                        "configured validation tolerance."
                    )

                else:

                    st.warning(
                        "Little's Law exceeds the configured "
                        "validation tolerance."
                    )

        except Exception as exc:

            st.error(
                f"Little's Law validation failed: {exc}"
            )


# ============================================================
# STAFFING & COST OPTIMISATION
if not inputs_available:
    st.warning(
        "Staffing optimisation requires valid arrival and service rates."
    )
else:
    # ============================================================

    st.divider()

    st.header("Staffing & Cost Optimisation")

    st.caption(
        "Identify the economically appropriate staffing level by "
        "balancing server capacity against the cost of customer waiting."
    )

    try:

        optimisation = staffing_cost_optimization(
            lam=lam,
            mu=mu,
            max_servers=int(max_servers),
            server_cost_per_hour=float(server_cost),
            waiting_cost_per_customer_hour=float(waiting_cost),
        )

        optimisation_df = optimisation["table"].copy()

        recommended_servers = optimisation[
            "recommended_servers"
        ]

        minimum_cost = optimisation[
            "minimum_total_cost"
        ]

        # --------------------------------------------------------
        # Decision summary
        # --------------------------------------------------------

        if recommended_servers is not None:

            opt1, opt2, opt3 = st.columns(3)

            opt1.metric(
                "Recommended Servers",
                str(recommended_servers),
            )

            opt2.metric(
                "Minimum Total Cost",
                f"₹{minimum_cost:,.2f}/hour",
            )

            recommended_row = optimisation_df[
                optimisation_df["servers"]
                == recommended_servers
            ].iloc[0]

            opt3.metric(
                "Recommended Utilisation",
                f"{recommended_row['rho']:.1%}",
            )

            st.success(
                f"QueueIQ recommends {recommended_servers} servers "
                f"under the current demand, service-rate and cost assumptions."
            )

        # --------------------------------------------------------
        # Cost table
        # --------------------------------------------------------

        st.subheader("Staffing Scenarios")

        cost_display = optimisation_df.copy()

        cost_display["rho"] = cost_display[
            "rho"
        ].map(
            lambda x: (
                f"{x:.1%}"
                if pd.notna(x)
                else "Unstable"
            )
        )

        cost_display["Lq"] = cost_display[
            "Lq"
        ].map(
            lambda x: (
                f"{x:.3f}"
                if pd.notna(x)
                else "—"
            )
        )

        cost_display["server_cost"] = cost_display[
            "server_cost"
        ].map(
            lambda x: f"₹{x:,.2f}"
        )

        cost_display["waiting_cost"] = cost_display[
            "waiting_cost"
        ].map(
            lambda x: (
                f"₹{x:,.2f}"
                if pd.notna(x) and x != float("inf")
                else "∞"
            )
        )

        cost_display["total_cost"] = cost_display[
            "total_cost"
        ].map(
            lambda x: (
                f"₹{x:,.2f}"
                if pd.notna(x) and x != float("inf")
                else "∞"
            )
        )

        cost_display = cost_display[
            [
                "servers",
                "rho",
                "Lq",
                "server_cost",
                "waiting_cost",
                "total_cost",
                "stable",
            ]
        ]

        cost_display.columns = [
            "Servers",
            "ρ",
            "Lq",
            "Server Cost / hr",
            "Waiting Cost / hr",
            "Total Cost / hr",
            "Stable",
        ]

        st.dataframe(
            cost_display,
            width="stretch",
            hide_index=True,
        )

        # --------------------------------------------------------
        # Cost comparison
        # --------------------------------------------------------

        st.subheader("Total Cost by Staffing Level")

        cost_chart = optimisation["table"].copy()

        cost_chart["Total Cost"] = cost_chart[
            "total_cost"
        ].replace(float("inf"), float("nan"))

        st.line_chart(
            cost_chart.set_index("servers")[
                "Total Cost"
            ],
            height=350,
        )

        # --------------------------------------------------------
        # Decision interpretation
        # --------------------------------------------------------

        if recommended_servers is not None:

            one_server = optimisation_df[
                optimisation_df["servers"] == 1
            ].iloc[0]

            recommended_row = optimisation_df[
                optimisation_df["servers"]
                == recommended_servers
            ].iloc[0]

            savings = (
                one_server["total_cost"]
                - recommended_row["total_cost"]
            )

            if savings > 0:

                st.info(
                    f"Moving from 1 server to {recommended_servers} "
                    f"reduces estimated total operating cost by "
                    f"₹{savings:,.2f} per hour under the current assumptions."
                )

            st.caption(
                "Decision rule: select the feasible staffing level "
                "with the minimum combined server and waiting cost. "
                "Management should review the cost assumptions before "
                "implementing staffing changes."
            )

    except Exception as exc:

        st.error(
            f"Staffing optimisation could not be completed: {exc}"
        )


# ============================================================
# DATA DIAGNOSTICS
# ============================================================

st.divider()

st.header("Data Diagnostics")

st.caption(
    "Assess dataset quality, derive queue parameters, and evaluate "
    "whether the observed arrival process supports the queueing assumptions."
)

if uploaded_df is None:

    st.info(
        "Upload a CSV or Excel dataset from the sidebar to activate "
        "data diagnostics."
    )

elif mapping is None:

    st.warning(
        "A dataset is loaded, but no column mapping is available."
    )

else:

    try:

        # --------------------------------------------------------
        # Dataset profile
        # --------------------------------------------------------

        diagnostics_profile = basic_profile(uploaded_df)

        st.subheader("Dataset Quality")

        p1, p2, p3, p4 = st.columns(4)

        p1.metric(
            "Rows",
            f"{diagnostics_profile['rows']:,}",
        )

        p2.metric(
            "Columns",
            f"{diagnostics_profile['columns']:,}",
        )

        p3.metric(
            "Missing Cells",
            f"{diagnostics_profile['missing_cells']:,}",
        )

        p4.metric(
            "Duplicate Rows",
            f"{diagnostics_profile['duplicate_rows']:,}",
        )

        # --------------------------------------------------------
        # Mapping summary
        # --------------------------------------------------------

        st.subheader("Queue Data Mapping")

        mapping_rows = []

        for field, column in mapping.items():

            mapping_rows.append(
                {
                    "QueueIQ Field": field.replace(
                        "_",
                        " ",
                    ).title(),
                    "Dataset Column": (
                        column
                        if column
                        else "Not mapped"
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(mapping_rows),
            width="stretch",
            hide_index=True,
        )

        # --------------------------------------------------------
        # Derived diagnostics
        # --------------------------------------------------------

        if data_diagnostics is not None:

            st.subheader("Derived Queue Parameters")

            d1, d2, d3, d4 = st.columns(4)

            estimated_lambda = data_diagnostics.get(
                "lambda_per_hour"
            )

            estimated_mu = data_diagnostics.get(
                "mu_per_hour"
            )

            interarrival_cv = data_diagnostics.get(
                "interarrival_cv"
            )

            service_cv = data_diagnostics.get(
                "service_cv"
            )

            d1.metric(
                "Estimated λ",
                (
                    f"{estimated_lambda:.3f}/hour"
                    if estimated_lambda is not None
                    else "Unavailable"
                ),
            )

            d2.metric(
                "Estimated μ",
                (
                    f"{estimated_mu:.3f}/hour"
                    if estimated_mu is not None
                    else "Unavailable"
                ),
            )

            d3.metric(
                "Inter-arrival CV",
                (
                    f"{interarrival_cv:.3f}"
                    if interarrival_cv is not None
                    else "Unavailable"
                ),
            )

            d4.metric(
                "Service-time CV",
                (
                    f"{service_cv:.3f}"
                    if service_cv is not None
                    else "Unavailable"
                ),
            )

            diagnostic_rows = [
                {
                    "Diagnostic": "Valid arrivals",
                    "Value": data_diagnostics.get(
                        "valid_arrivals",
                        0,
                    ),
                },
                {
                    "Diagnostic": "Valid waiting times",
                    "Value": data_diagnostics.get(
                        "valid_waiting_times",
                        0,
                    ),
                },
                {
                    "Diagnostic": "Valid service times",
                    "Value": data_diagnostics.get(
                        "valid_service_times",
                        0,
                    ),
                },
                {
                    "Diagnostic": "Mean inter-arrival time",
                    "Value": (
                        f"{data_diagnostics['mean_interarrival_seconds']:.2f} sec"
                        if data_diagnostics.get(
                            "mean_interarrival_seconds"
                        ) is not None
                        else "Unavailable"
                    ),
                },
                {
                    "Diagnostic": "Mean waiting time",
                    "Value": (
                        f"{data_diagnostics['mean_waiting_seconds'] / 60:.2f} min"
                        if data_diagnostics.get(
                            "mean_waiting_seconds"
                        ) is not None
                        else "Unavailable"
                    ),
                },
                {
                    "Diagnostic": "Mean service time",
                    "Value": (
                        f"{data_diagnostics['mean_service_seconds'] / 60:.2f} min"
                        if data_diagnostics.get(
                            "mean_service_seconds"
                        ) is not None
                        else "Unavailable"
                    ),
                },
            ]

            st.dataframe(
                pd.DataFrame(diagnostic_rows),
                width="stretch",
                hide_index=True,
            )

        # --------------------------------------------------------
        # Arrival-process diagnostic
        # --------------------------------------------------------

        st.subheader(
            "Arrival Process Diagnostic"
        )

        poisson_result = poisson_chi_square_test(
            uploaded_df,
            mapping,
        )

        q1, q2, q3, q4 = st.columns(4)

        q1.metric(
            "Sample Size",
            f"{poisson_result['sample_size']:,}",
        )

        q2.metric(
            "Chi-square",
            (
                f"{poisson_result['chi_square']:.3f}"
                if poisson_result["chi_square"] is not None
                else "Unavailable"
            ),
        )

        q3.metric(
            "Degrees of Freedom",
            (
                str(poisson_result["degrees_of_freedom"])
                if poisson_result["degrees_of_freedom"] is not None
                else "Unavailable"
            ),
        )

        q4.metric(
            "p-value",
            (
                f"{poisson_result['p_value']:.4f}"
                if poisson_result["p_value"] is not None
                else "Unavailable"
            ),
        )

        decision = poisson_result.get(
            "decision"
        )

        interpretation = poisson_result.get(
            "interpretation"
        )

        if decision == "Do not reject exponential inter-arrival distribution":

            st.success(
                decision
            )

        elif decision == "Reject exponential inter-arrival distribution":

            st.warning(
                decision
            )

        else:

            st.info(
                decision
                if decision
                else "Arrival-process diagnostic unavailable."
            )

        if interpretation:

            st.info(
                interpretation
            )

        # --------------------------------------------------------
        # Data readiness assessment
        # --------------------------------------------------------

        st.subheader("Queue Modelling Readiness")

        readiness_issues = []

        if diagnostics_profile["missing_cells"] > 0:

            readiness_issues.append(
                f"{diagnostics_profile['missing_cells']:,} missing cells detected."
            )

        if diagnostics_profile["duplicate_rows"] > 0:

            readiness_issues.append(
                f"{diagnostics_profile['duplicate_rows']:,} duplicate rows detected."
            )

        if data_diagnostics is None:

            readiness_issues.append(
                "Queue parameters could not be derived."
            )

        else:

            if data_diagnostics.get(
                "lambda_per_hour"
            ) is None:

                readiness_issues.append(
                    "Arrival rate λ could not be estimated."
                )

            if data_diagnostics.get(
                "mu_per_hour"
            ) is None:

                readiness_issues.append(
                    "Service rate μ could not be estimated."
                )

            if data_diagnostics.get(
                "valid_arrivals",
                0,
            ) < 40:

                readiness_issues.append(
                    "Fewer than 40 valid arrival observations are available "
                    "for the current arrival-process diagnostic."
                )

        if readiness_issues:

            st.warning(
                "Review the following data-quality considerations:"
            )

            for issue in readiness_issues:

                st.write(
                    f"- {issue}"
                )

        else:

            st.success(
                "The dataset contains sufficient mapped information "
                "for the current queueing diagnostics."
            )

        st.caption(
            "The arrival-process test evaluates consistency with an "
            "exponential inter-arrival distribution. It supports, but "
            "does not prove, a Poisson arrival assumption."
        )

    except Exception as exc:

        st.error(
            f"Data diagnostics could not be completed: {exc}"
        )
