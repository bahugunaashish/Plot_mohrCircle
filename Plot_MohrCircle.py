# ══════════════════════════════════════════════════════════════════════════════
# By Ashish Bahuguna, email: bahugunaashish92@gmail.com 
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt

# ============================================================
# MOHR-COULOMB ANALYSER — Streamlit version
# Converted from the original Tkinter application.
# ============================================================

st.set_page_config(
    page_title="Mohr-Coulomb Analyser",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- Theme -----------------------------
BG = "#f4f4f4"
PANEL = "#ffffff"
ACCENT = "#2563eb"
ACCENT2 = "#dc2626"
TEXT = "#111111"
MUTED = "#555555"
ERROR = "#dc2626"

CIRCLE_COLORS = [
    "#4f9cf9", "#4fcf8f", "#f97b4f",
    "#c97bf9", "#f9d44f", "#f94f8a",
    "#4ff9e2", "#f9a84f",
]

# --------------------- Geometry helpers ---------------------

def compute_envelope(centers, radii):
    """Least-squares common tangent: r_i = A + B*cx_i."""
    if len(centers) < 2:
        return None, None

    X = np.column_stack([np.ones_like(centers), centers])
    (A, B), *_ = lstsq(X, radii, rcond=None)
    B = np.clip(B, -0.9999, 0.9999)

    phi_rad = np.arcsin(B)
    c_prime = A / np.cos(phi_rad)
    return phi_rad, c_prime


def tangent_point(cx, r, phi_rad):
    tx = cx - r * np.sin(phi_rad)
    ty = r * np.cos(phi_rad)
    return tx, ty


def shear_strength(c_prime, phi_rad, sigma_eff):
    return c_prime + sigma_eff * np.tan(phi_rad)


# ----------------------- Session state ----------------------

EXAMPLE_DATA = [
    {"sigma3": 100.0, "delta": 170.0, "pore": -15.0},
    {"sigma3": 200.0, "delta": 260.0, "pore": -40.0},
    {"sigma3": 300.0, "delta": 360.0, "pore": -80.0},
]


def initialise():
    if "tests" not in st.session_state:
        st.session_state.tests = [dict(x) for x in EXAMPLE_DATA]

    if "gamma_sat" not in st.session_state:
        st.session_state.gamma_sat = 19.0
    if "gamma_w" not in st.session_state:
        st.session_state.gamma_w = 9.81
    if "depth" not in st.session_state:
        st.session_state.depth = 2.0


def load_example():
    st.session_state.tests = [dict(x) for x in EXAMPLE_DATA]
    st.session_state.gamma_sat = 19.0
    st.session_state.gamma_w = 9.81
    st.session_state.depth = 2.0


def clear_all():
    st.session_state.tests = [
        {"sigma3": None, "delta": None, "pore": None}
    ]


initialise()

# --------------------------- CSS -----------------------------

st.markdown(
    """
    <style>
    .main {
        background-color: #f4f4f4;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
    }
    .app-title {
        background: #2563eb;
        color: white;
        padding: 0.65rem 1rem;
        border-radius: 0.45rem;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    .result-card {
        background: white;
        border: 1px solid #d1d1d1;
        border-radius: 0.45rem;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
    }
    .result-label {
        color: #555555;
        font-size: 0.85rem;
    }
    .result-value {
        color: #2563eb;
        font-size: 1.25rem;
        font-weight: 700;
    }
    .small-muted {
        color: #555555;
        font-size: 0.82rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------- Header --------------------------

st.markdown(
    '<div class="app-title">⬡ MOHR-COULOMB ANALYSER</div>',
    unsafe_allow_html=True,
)

# ---------------------- Main calculation --------------------

sigma3_list = []
delta_list = []
pore_list = []

left, right = st.columns([0.36, 0.64], gap="large")

with left:
    st.subheader("Soil Parameters")

    p1, p2 = st.columns(2)
    with p1:
        st.session_state.gamma_sat = st.number_input(
            "γ_sat (kN/m³)",
            value=float(st.session_state.gamma_sat),
            step=0.1,
            format="%.3f",
        )
    with p2:
        st.session_state.gamma_w = st.number_input(
            "γ_w (kN/m³)",
            value=float(st.session_state.gamma_w),
            step=0.01,
            format="%.3f",
        )

    st.session_state.depth = st.number_input(
        "Depth z (m)",
        value=float(st.session_state.depth),
        min_value=0.0,
        step=0.1,
        format="%.3f",
    )

    st.divider()
    st.subheader("CU Triaxial Test Data")

    # Header
    h1, h2, h3, h4, h5 = st.columns([0.07, 0.27, 0.27, 0.27, 0.12])
    h1.markdown("**#**")
    h2.markdown("**σ₃ (kPa)**")
    h3.markdown("**Δσf (kPa)**")
    h4.markdown("**u (kPa)**")
    h5.markdown("**Delete**")

    rows_to_delete = []

    for i, row in enumerate(st.session_state.tests):
        c1, c2, c3, c4, c5 = st.columns([0.07, 0.27, 0.27, 0.27, 0.12])

        with c1:
            st.write(f"**{i + 1}**")

        with c2:
            val = st.number_input(
                f"sigma3_{i}",
                value=float(row["sigma3"]) if row["sigma3"] is not None else None,
                step=1.0,
                format="%.3f",
                label_visibility="collapsed",
                key=f"sigma3_{i}",
            )
            st.session_state.tests[i]["sigma3"] = val

        with c3:
            val = st.number_input(
                f"delta_{i}",
                value=float(row["delta"]) if row["delta"] is not None else None,
                step=1.0,
                format="%.3f",
                label_visibility="collapsed",
                key=f"delta_{i}",
            )
            st.session_state.tests[i]["delta"] = val

        with c4:
            val = st.number_input(
                f"pore_{i}",
                value=float(row["pore"]) if row["pore"] is not None else None,
                step=1.0,
                format="%.3f",
                label_visibility="collapsed",
                key=f"pore_{i}",
            )
            st.session_state.tests[i]["pore"] = val

        with c5:
            if st.button("✕", key=f"delete_{i}", help=f"Delete test {i + 1}"):
                rows_to_delete.append(i)

    if rows_to_delete:
        for i in reversed(rows_to_delete):
            del st.session_state.tests[i]
        if not st.session_state.tests:
            st.session_state.tests = [
                {"sigma3": None, "delta": None, "pore": None}
            ]
        st.rerun()

    b1, b2, b3 = st.columns(3)

    with b1:
        if st.button("+ ADD ROW", use_container_width=True, type="primary"):
            st.session_state.tests.append(
                {"sigma3": None, "delta": None, "pore": None}
            )
            st.rerun()

    with b2:
        if st.button("CLEAR ALL", use_container_width=True):
            clear_all()
            st.rerun()

    with b3:
        if st.button("EXAMPLE", use_container_width=True):
            load_example()
            st.rerun()

    st.divider()

    # Parse valid rows
    for row in st.session_state.tests:
        try:
            if (
                row["sigma3"] is not None
                and row["delta"] is not None
                and row["pore"] is not None
            ):
                sigma3_list.append(float(row["sigma3"]))
                delta_list.append(float(row["delta"]))
                pore_list.append(float(row["pore"]))
        except (ValueError, TypeError):
            pass

    sigma3_arr = np.array(sigma3_list, dtype=float)
    delta_arr = np.array(delta_list, dtype=float)
    pore_arr = np.array(pore_list, dtype=float)

    # ---------------- Results ----------------

    st.subheader("Results")

    result_phi = "—"
    result_c = "—"
    result_tau = "—"
    result_info = ""

    n = len(sigma3_arr)

    if n >= 2:
        sigma3_eff = sigma3_arr - pore_arr
        sigma1_eff = sigma3_arr + delta_arr - pore_arr
        centers = (sigma1_eff + sigma3_eff) / 2
        radii = (sigma1_eff - sigma3_eff) / 2

        phi_rad, c_prime = compute_envelope(centers, radii)

        if (
            phi_rad is not None
            and c_prime is not None
            and np.isfinite(phi_rad)
            and np.isfinite(c_prime)
        ):
            phi_deg = np.degrees(phi_rad)
            result_phi = f"{phi_deg:.2f}°"
            result_c = f"{c_prime:.2f} kPa"

            try:
                gs = float(st.session_state.gamma_sat)
                gw = float(st.session_state.gamma_w)
                z = float(st.session_state.depth)

                s_eff = (gs - gw) * z
                tau_f = shear_strength(c_prime, phi_rad, s_eff)

                if np.isfinite(tau_f):
                    result_tau = f"{tau_f:.2f} kPa"
                    result_info = f"σ'v = {s_eff:.2f} kPa at z = {z:.2f} m"
                else:
                    result_info = "Check soil parameters."
            except (ValueError, ZeroDivisionError):
                result_info = "Check soil parameters."
        else:
            result_info = "Could not fit envelope."

    elif n == 1:
        result_info = "Add ≥ 2 tests for envelope."
    else:
        result_info = "Enter at least 1 complete test row."

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("φ′", result_phi)
    with r2:
        st.metric("c′", result_c)
    with r3:
        st.metric("τf (mid-layer)", result_tau)

    if result_info:
        st.caption(result_info)


# ---------------------------- Plot ---------------------------

with right:
    st.subheader("Mohr-Coulomb Failure Envelope — CU Triaxial Tests")

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.axhline(0, color="#d1d1d1", linewidth=0.8)
    ax.axvline(0, color="#d1d1d1", linewidth=0.8)

    ax.set_xlabel("Effective Normal Stress  σ′ (kPa)", fontsize=10)
    ax.set_ylabel("Shear Stress  τ (kPa)", fontsize=10)
    ax.tick_params(labelsize=8)

    n = len(sigma3_arr)

    if n == 0:
        ax.set_title("No valid test data", color=MUTED, fontsize=11)
        ax.set_xlim(-20, 300)
        ax.set_ylim(-5, 100)

    else:
        sigma3_eff = sigma3_arr - pore_arr
        sigma1_eff = sigma3_arr + delta_arr - pore_arr
        centers = (sigma1_eff + sigma3_eff) / 2
        radii = (sigma1_eff - sigma3_eff) / 2

        # Protect plot limits against invalid/negative test geometry.
        finite_values = np.concatenate([sigma1_eff, radii])
        finite_values = finite_values[np.isfinite(finite_values)]

        if len(finite_values) == 0:
            max_x = 300.0
            max_y = 100.0
        else:
            max_x = max(float(np.max(sigma1_eff)) * 1.12, 20.0)
            max_y = max(float(np.max(radii)) * 1.25, 10.0)

        ax.set_xlim(-max_x * 0.04, max_x)
        ax.set_ylim(-max_y * 0.04, max_y)

        theta = np.linspace(0, np.pi, 400)

        # Draw Mohr circles
        for i, (cx, r) in enumerate(zip(centers, radii)):
            col = CIRCLE_COLORS[i % len(CIRCLE_COLORS)]
            x = cx + r * np.cos(theta)
            y = r * np.sin(theta)

            ax.plot(
                x,
                y,
                color=col,
                linewidth=1.8,
                label=(
                    f"Test {i + 1}  "
                    f"σ₃′={sigma3_eff[i]:.0f}, "
                    f"σ₁′={sigma1_eff[i]:.0f} kPa"
                ),
                zorder=3,
            )

            ax.plot(
                [cx - r, cx + r],
                [0, 0],
                color=col,
                linewidth=0.5,
                linestyle="--",
                alpha=0.35,
                zorder=2,
            )

        # Failure envelope
        if n >= 2:
            phi_rad, c_prime = compute_envelope(centers, radii)

            if (
                phi_rad is not None
                and c_prime is not None
                and np.isfinite(phi_rad)
                and np.isfinite(c_prime)
            ):
                phi_deg = np.degrees(phi_rad)

                # Tangency points
                for i, (cx, r) in enumerate(zip(centers, radii)):
                    col = CIRCLE_COLORS[i % len(CIRCLE_COLORS)]
                    tx, ty = tangent_point(cx, r, phi_rad)

                    ax.vlines(
                        tx, 0, ty,
                        linestyles="dotted",
                        colors=col,
                        linewidth=0.8,
                        alpha=0.7,
                    )
                    ax.plot(
                        tx, ty, "o",
                        color=col,
                        markersize=6,
                        markeredgecolor=BG,
                        markeredgewidth=0.8,
                        zorder=5,
                    )

                # Envelope
                sigma_env = np.array([-max_x * 0.04, max_x])
                tau_env = c_prime + sigma_env * np.tan(phi_rad)

                ax.plot(
                    sigma_env,
                    tau_env,
                    color=ACCENT2,
                    linewidth=2,
                    label=(
                        f"Failure envelope  "
                        f"c′={c_prime:.1f} kPa, φ′={phi_deg:.1f}°"
                    ),
                    zorder=6,
                )

                # c′ intercept
                if 0 <= c_prime <= max_y:
                    ax.plot(
                        0, c_prime, "^",
                        color=ACCENT2,
                        markersize=8,
                        zorder=7,
                    )
                    ax.annotate(
                        f"c′={c_prime:.1f} kPa",
                        xy=(0, c_prime),
                        xytext=(
                            max_x * 0.06,
                            c_prime + max_y * 0.06,
                        ),
                        color=ACCENT2,
                        fontsize=8,
                        arrowprops=dict(
                            arrowstyle="->",
                            color=ACCENT2,
                            lw=1,
                        ),
                    )

                # φ′ arc
                arc_r = min(max_x * 0.09, 60)
                arc_th = np.linspace(0, phi_rad, 120)

                ax.plot(
                    arc_r * np.cos(arc_th),
                    c_prime + arc_r * np.sin(arc_th),
                    color=MUTED,
                    linewidth=0.8,
                )
                ax.text(
                    arc_r * 0.55,
                    c_prime + arc_r * 0.28,
                    f"φ′={phi_deg:.1f}°",
                    color=MUTED,
                    fontsize=8,
                )

                # Mid-layer shear strength
                try:
                    gs = float(st.session_state.gamma_sat)
                    gw = float(st.session_state.gamma_w)
                    z = float(st.session_state.depth)

                    s_eff = (gs - gw) * z
                    tau_f = shear_strength(c_prime, phi_rad, s_eff)

                    if np.isfinite(tau_f):
                        ax.plot(
                            s_eff, tau_f, "*",
                            color=ERROR,
                            markersize=13,
                            zorder=7,
                            label=f"Mid-layer τf = {tau_f:.1f} kPa",
                        )
                        ax.vlines(
                            s_eff, 0, tau_f,
                            linestyles="dotted",
                            colors=ERROR,
                            linewidth=1,
                        )
                        ax.hlines(
                            tau_f, 0, s_eff,
                            linestyles="dotted",
                            colors=ERROR,
                            linewidth=1,
                        )
                        ax.annotate(
                            f"τf={tau_f:.1f} kPa",
                            xy=(s_eff, tau_f),
                            xytext=(
                                s_eff + max_x * 0.06,
                                tau_f + max_y * 0.05,
                            ),
                            color=ERROR,
                            fontsize=8,
                            arrowprops=dict(
                                arrowstyle="->",
                                color=ERROR,
                                lw=1,
                            ),
                        )
                except (ValueError, ZeroDivisionError):
                    pass

        elif n == 1:
            pass

    ax.set_aspect("equal", adjustable="datalim")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        leg = ax.legend(
            loc="upper left",
            fontsize=7.5,
            framealpha=0.85,
            edgecolor="#cccccc",
        )
        for t in leg.get_texts():
            t.set_color(TEXT)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

#st.caption(
 #   "Streamlit version of the original Mohr-Coulomb Analyser. "
 #   "Calculations retain the original least-squares envelope method."
)


