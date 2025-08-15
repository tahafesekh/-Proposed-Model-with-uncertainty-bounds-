import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

# ===== CSS Styles for Streamlit interface and DataFrame =====
st.markdown("""
    <style>
        .stApp { background-color: #eeeeee !important; color: black !important; }
        .stMarkdown, .stTitle, .stHeader, .stSubheader, .stCaption, .stTextInput > label,
        .stNumberInput > label, .stSelectbox > label, .stInfo, .stAlert, .stSuccess, .stError, .stButton > button {
            color: black !important;
        }
        [data-testid="stSidebar"] { background-color: white !important; }
        .stButton > button { color: black !important; border: 1px solid #333; background: #fff !important; }
        .stAlert, .stInfo, .stSuccess, .stError {
            background-color: #e3f2fd !important; color: #222 !important; border-left: 5px solid #2196f3 !important;
            border-radius: 6px !important; box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        }
        .stInfo > div, .stAlert > div, .stSuccess > div, .stError > div {
            color: #222 !important; background-color: #e3f2fd !important;
        }
        .special-bold-box {
            background-color: #d2ffd2 !important; color: #185d18 !important; border-left: 5px solid #219653 !important;
            border-radius: 7px !important; box-shadow: 0 2px 8px rgba(0,0,0,0.03);
            font-weight: bold !important; padding: 0.8em 0.7em !important; margin-bottom: 8px !important;
        }
        div[data-testid="stDataFrameContainer"] { background-color: white !important; color: black !important; }
        .css-1lcbmhc.e1fqkh3o3 { background-color: white !important; color: black !important; }
        [data-testid="stDownloadButton"] > button {
            color: white !important; background-color: #1976d2 !important; border-radius: 6px !important;
            border: 2px solid #1976d2 !important; font-weight: bold !important; padding: 0.5em 1.5em !important;
            font-size: 1.05em !important; box-shadow: 0 2px 8px rgba(25, 118, 210, 0.12); transition: 0.2s;
        }
        [data-testid="stDownloadButton"] > button:hover {
            background-color: #1565c0 !important; border: 2px solid #1565c0 !important; color: #fff !important; cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* Force the whole app widgets into light mode */
:root {
    color-scheme: light !important;
}

/* ===== Checkbox label ===== */
div[data-testid="stCheckbox"] p, 
div[data-testid="stCheckbox"] label, 
[data-baseweb="checkbox"] label, 
[data-baseweb="checkbox"] span {
    color: #000 !important;     /* Force black text */
    opacity: 1 !important;
}

/* ===== Native checkbox input ===== */
div[data-testid="stCheckbox"] input[type="checkbox"] {
    accent-color: #1976d2 !important;  /* Blue highlight */
    background-color: #fff !important; /* White box */
    border: 1px solid #333 !important;
}

/* ===== BaseWeb custom checkbox (most Streamlit versions) ===== */
[data-baseweb="checkbox"] > label > div:first-child {
    background-color: #fff !important;  /* White background */
    border: 1px solid #333 !important;
    box-shadow: none !important;
}
[data-baseweb="checkbox"] input:checked + div {
    background-color: #1976d2 !important;  /* Blue when checked */
    border-color: #1976d2 !important;
}
[data-baseweb="checkbox"] svg {
    fill: #fff !important;                /* White checkmark */
}
</style>
""", unsafe_allow_html=True)


st.title("Calculation of the Maximum Deflection Using the Proposed Model")
st.markdown("by Taha A.Fesekh, Ehab M. Lotfy, Erfan Abdel-Latif, Nady M. Abdel-Fattah, and Abdel-Rahman M. Naguib")

# =========================
# Helpers
# =========================
def compute_section_and_deflection(params):
    """Run the original deterministic calculations and return a dict of key results."""
    (section_type, Pa, X, L, fc, Ef, Af, rho_f, rho_fb, d, d_prime, b,
     y1, B, y2, t) = params

    # Calculations (original)
    Ec = 4700 * (fc ** 0.5)
    nf = Ef / Ec if Ec != 0 else 0
    beta_d = min(0.06 * (rho_f / rho_fb), 0.50) if rho_fb != 0 else 0.50
    Xd_ratio = X / d if d != 0 else 0
    if Xd_ratio <= 2.8:
        lambda_e = min(max(0.085 * Xd_ratio + 0.00813 * d_prime + 0.44, 0.80), 0.95)
    else:
        lambda_e = min(max(-0.0074 * Xd_ratio + 0.0094 * d_prime + 0.30, 0.55), 0.95)

    if section_type == "T-section":
        A1 = b * y1
        A2 = B * y2
        At = A1 + A2 + (nf - 1) * Af
        yt = (A1 * (y1 / 2) + A2 * (y1 + y2 / 2) + (nf - 1) * Af * d_prime) / At if At != 0 else 0
        Ig = (b * y1 ** 3) / 12 + A1 * (y1 / 2 - yt) ** 2 + (B * y2 ** 3) / 12 + A2 * ((y1 + y2 / 2) - yt) ** 2 + ((nf - 1) * Af) * (yt - d_prime) ** 2
    else:
        A1 = b * t
        A2 = 0
        At = b * t + (nf - 1) * Af
        yt = ((b * t ** 2) / 2 + (nf - 1) * Af * d_prime) / At if At != 0 else 0
        Ig = (b * t ** 3) / 12 + b * t * (t / 2 - yt) ** 2 + ((nf - 1) * Af) * (yt - d_prime) ** 2

    Ma = (Pa * 1000) / 2 * X  # N.mm
    fr = 0.62 * math.sqrt(fc)
    Mcr = fr * Ig / yt if yt != 0 else 0

    # Neutral axis and Icr
    if section_type == "T-section":
        a = B / 2
    else:
        a = b / 2
    b_quad = nf * Af
    c = -(nf * Af) * d
    delta = b_quad ** 2 - 4 * a * c
    if delta < 0 or a == 0:
        Z = 0
    else:
        Z = (-b_quad + math.sqrt(delta)) / (2 * a)

    if section_type == "T-section":
        Icr = B * Z ** 3 / 3 + (nf * Af) * (d - Z) ** 2
    else:
        Icr = b * Z ** 3 / 3 + (nf * Af) * (d - Z) ** 2

    # Effective I
    if Ma < Mcr:
        Ie = Ig
    else:
        ratio = (Mcr / Ma) ** 3 if Ma > 0 else 0
        Ie = beta_d * ratio * Ig + (min(max(lambda_e, 0.0), 0.95)) * (1 - ratio) * Icr
        Ie = min(Ie, Ig)

    # Deflection
    if Ec != 0 and Ie != 0:
        delta_max = (Pa * 1000 * X) / (48 * Ec * Ie) * (3 * L ** 2 - 4 * X ** 2)
    else:
        delta_max = 0.0

    return {
        "A1": A1, "A2": A2, "Ec": Ec, "nf": nf, "beta_d": beta_d, "lambda_e": lambda_e,
        "yt": yt, "At": At, "Ig": Ig, "Ma": Ma, "fr": fr, "Mcr": Mcr,
        "Z": Z, "Icr": Icr, "Ie": Ie, "delta_max": delta_max
    }

def ci_from_samples(samples, conf=0.95):
    """Return mean, lower, upper for a two-sided percentile CI."""
    if len(samples) == 0:
        return 0.0, 0.0, 0.0
    s = np.asarray(samples)
    mean = float(np.mean(s))
    alpha = 1 - conf
    low = float(np.percentile(s, 100 * (alpha / 2)))
    high = float(np.percentile(s, 100 * (1 - alpha / 2)))
    return mean, low, high

def draw_positive_normal(mu, rel_sigma, size=1, min_val=1e-9):
    """Sample normal with relative std; truncate to positive."""
    sigma = abs(mu) * rel_sigma
    samp = np.random.normal(mu, sigma, size=size)
    return np.clip(samp, min_val, None)

# =========================
# Inputs
# =========================
section_type = st.selectbox("Section Type", ["T-section", "R-section"])

# Default values
if section_type == "T-section":
    default_Pa, default_X, default_L = 118.0, 600.0, 1800.0
    default_fc, default_Ef, default_Af = 26.80, 50000.0, 157.0
    default_rho_f, default_rho_fb = 0.35, 0.23
    default_d, default_d_prime = 225.0, 25.0
    default_b, default_y1, default_B, default_y2 = 200.0, 175.0, 500.0, 75.0
    default_t = None
else:
    default_Pa, default_X, default_L = 107.0, 600.0, 1800.0
    default_fc, default_Ef, default_Af = 26.80, 50000.0, 157.0
    default_rho_f, default_rho_fb = 0.35, 0.23
    default_d, default_d_prime = 225.0, 25.0
    default_b, default_t = 200.0, 250.0
    default_y1 = default_B = default_y2 = None

st.subheader("Input Data")
col1, col2, col3, col4 = st.columns(4)
with col1: Pa = st.number_input("Applied Load (Pa, kN)", value=default_Pa)
with col2: X = st.number_input("Flexural-shear span X (mm)", value=default_X)
with col3: L = st.number_input("Beam Length L (mm)", value=default_L)
with col4: fc = st.number_input("Concrete compressive strength f'c (MPa)", value=default_fc)

col5, col6, col7, col8 = st.columns(4)
with col5: Ef = st.number_input("Modulus of elasticity of FRP Ef (MPa)", value=default_Ef)
with col6: Af = st.number_input("Area of FRP reinforcement Af (mm2)", value=default_Af)
with col7: rho_f = st.number_input("Reinforcement ratio rho_f", value=default_rho_f)
with col8: rho_fb = st.number_input("Balanced reinforcement ratio rho_fb", value=default_rho_fb)

col9, col10 = st.columns(2)
with col9: d = st.number_input("Effective depth d (mm)", value=default_d)
with col10: d_prime = st.number_input("Concrete Cover d' (mm)", value=default_d_prime)

if section_type == "T-section":
    colT1, colT2, colT3, colT4 = st.columns(4)
    with colT1: b = st.number_input("Width of flange b (mm)", value=200.0)
    with colT2: y1 = st.number_input("Height of flange y1 (mm)", value=175.0)
    with colT3: B = st.number_input("Width of web B (mm)", value=500.0)
    with colT4: y2 = st.number_input("Height of web y2 (mm)", value=75.0)
    t = None
else:
    colR1, colR2 = st.columns(2)
    with colR1: b = st.number_input("Width b (mm)", value=200.0)
    with colR2: t = st.number_input("Height t (mm)", value=250.0)
    y1 = B = y2 = None

# =========================
# Uncertainty controls
# =========================
st.subheader("Uncertainty (Optional)")
enable_unc = st.checkbox("Enable uncertainty analysis (Monte Carlo)", value=False)
if enable_unc:
    colu1, colu2, colu3, colu4 = st.columns(4)
    with colu1: u_Pa = st.number_input("Pa uncertainty ±% (rel. std)", value=5.0, min_value=0.0, step=0.5)
    with colu2: u_fc = st.number_input("f'c uncertainty ±% (rel. std)", value=7.5, min_value=0.0, step=0.5)
    with colu3: u_Ef = st.number_input("Ef uncertainty ±% (rel. std)", value=5.0, min_value=0.0, step=0.5)
    with colu4: u_Af = st.number_input("Af uncertainty ±% (rel. std)", value=2.0, min_value=0.0, step=0.5)

    colu5, colu6 = st.columns(2)
    with colu5: Nsim = st.number_input("Number of simulations", value=2000, min_value=100, step=100)
    with colu6: conf = st.selectbox("Confidence level", [0.80, 0.90, 0.95, 0.99], index=2)
else:
    u_Pa = u_fc = u_Ef = u_Af = 0.0
    Nsim = 0
    conf = 0.95

# =========================
# Deterministic computation
# =========================
params = (section_type, Pa, X, L, fc, Ef, Af, rho_f, rho_fb, d, d_prime, b,
          y1 if y1 is not None else 0.0,
          B if B is not None else 0.0,
          y2 if y2 is not None else 0.0,
          t if t is not None else 0.0)

det = compute_section_and_deflection(params)

# =========================
# UI - Run
# =========================
if st.button("Run"):
    # Show deterministic results (original boxes)
    results = []
    if section_type == "T-section":
        A1 = b * y1; A2 = B * y2
        results.append(f"Area of web A1 = {A1:.2f} mm²")
        results.append(f"Area of flange A2 = {A2:.2f} mm²")
    else:
        A1 = b * t
        results.append(f"Area = {A1:.2f} mm²")
    results.extend([
        f"Elastic modulus of concrete Ec = {det['Ec']:.2f} MPa",
        f"Modular ratio nf (Ef/Ec) = {det['nf']:.3f} (calculated)",
        f"Reduction coefficient βd = {det['beta_d']:.3f}",
        f"Reduction factor λe = {det['lambda_e']:.3f}",
        f"Neutral axis depth y_t = {det['yt']:.3f} mm",
        f"Equivalent area A_t = {det['At']:.3f} mm²",
        f"Ig (Gross moment of inertia) = {det['Ig']:.3f} mm⁴",
        f"Maximum moment Ma = {det['Ma']:.2f} N·mm",
        f"Modulus of rupture fr = {det['fr']:.3f} MPa",
        f"Cracking moment Mcr = {det['Mcr']:.2f} N·mm",
        f"Compression zone depth Z = {det['Z']:.3f} mm",
        f"Icr (cracked moment of inertia) = {det['Icr']:.3f} mm⁴",
        f"Effective moment of inertia Ie = {det['Ie']:.3f} mm⁴",
        f"Maximum Deflection δ_max = {det['delta_max']:.5f} mm"
    ])

    cols_per_row = 4
    idx_Ie = len(results) - 2
    idx_defl = len(results) - 1
    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for idx, col in enumerate(cols):
            res_idx = i + idx
            if res_idx < len(results):
                result = results[res_idx]
                if res_idx in (idx_Ie, idx_defl):
                    col.markdown(f'<div class="special-bold-box">{result}</div>', unsafe_allow_html=True)
                else:
                    col.info(result)
    st.success("Done")

    # =========================
    # Uncertainty analysis (Monte Carlo)
    # =========================
    if enable_unc:
        # Prepare samples
        Pa_s = draw_positive_normal(Pa, u_Pa/100.0, size=Nsim)
        fc_s = draw_positive_normal(fc, u_fc/100.0, size=Nsim)
        Ef_s = draw_positive_normal(Ef, u_Ef/100.0, size=Nsim)
        Af_s = draw_positive_normal(Af, u_Af/100.0, size=Nsim)

        delta_samples = []
        Ie_samples = []

        for i in range(Nsim):
            p = (section_type, float(Pa_s[i]), X, L, float(fc_s[i]), float(Ef_s[i]), float(Af_s[i]),
                 rho_f, rho_fb, d, d_prime, b,
                 y1 if y1 is not None else 0.0,
                 B if B is not None else 0.0,
                 y2 if y2 is not None else 0.0,
                 t if t is not None else 0.0)
            out = compute_section_and_deflection(p)
            delta_samples.append(out["delta_max"])
            Ie_samples.append(out["Ie"])

        # Confidence intervals
        d_mean, d_low, d_high = ci_from_samples(delta_samples, conf=float(conf))
        i_mean, i_low, i_high = ci_from_samples(Ie_samples, conf=float(conf))

        # Show summary boxes
        st.subheader("Uncertainty Summary")
        coluA, coluB = st.columns(2)
        with coluA:
            st.markdown(
                f'<div class="special-bold-box">δ_max = {d_mean:.5f} mm '
                f'(CI {int(conf*100)}%: {d_low:.5f} – {d_high:.5f})</div>',
                unsafe_allow_html=True
            )
        with coluB:
            st.markdown(
                f'<div class="special-bold-box">Ie = {i_mean:.3f} mm⁴ '
                f'(CI {int(conf*100)}%: {i_low:.3f} – {i_high:.3f})</div>',
                unsafe_allow_html=True
            )

        # Histogram of delta_max samples
        fig_hist, axh = plt.subplots(figsize=(7, 4))
        axh.hist(delta_samples, bins=40, edgecolor='black')
        axh.set_xlabel("δ_max (mm)")
        axh.set_ylabel("Frequency")
        axh.set_title(f"Monte Carlo Distribution of δ_max ({Nsim} sims, CI {int(conf*100)}%)")
        st.pyplot(fig_hist)

        # Download samples as CSV
        df_sims = pd.DataFrame({
            "delta_max_mm": delta_samples,
            "Ie_mm4": Ie_samples,
            "Pa_kN": Pa_s,
            "fc_MPa": fc_s,
            "Ef_MPa": Ef_s,
            "Af_mm2": Af_s
        })
        csv_bytes = df_sims.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Monte Carlo samples (CSV)",
            data=csv_bytes,
            file_name="uncertainty_samples.csv",
            mime="text/csv"
        )

    # =========================
    # Load–Deflection curve (deterministic as before)
    # =========================
    smooth_percentages = np.linspace(0, 100, 51)
    smooth_loads = Pa * (smooth_percentages / 100)
    smooth_deflections = []
    Mcr = det["Mcr"]; Ig = det["Ig"]; beta_d = det["beta_d"]; lambda_e = det["lambda_e"]; Icr = det["Icr"]
    Ec = det["Ec"]

    for load in smooth_loads:
        Ma_i = (load * 1000) / 2 * X
        smoothing_range = 0.1 * Mcr if Mcr != 0 else 1
        x = (Ma_i - Mcr) / smoothing_range if smoothing_range != 0 else 0
        w = 0.5 * (1 + np.tanh(x))
        if Ma_i < 0.01:
            Ie_i = Ig
        else:
            ratio_i = (Mcr / Ma_i) ** 3 if Ma_i > 0 else 1
            Ie_uncracked = Ig
            Ie_cracked = beta_d * ratio_i * Ig + lambda_e * (1 - ratio_i) * Icr
            Ie_i = (1 - w) * Ie_uncracked + w * Ie_cracked
            Ie_i = min(Ie_i, Ig)
        if Ec != 0 and Ie_i != 0:
            delta_i = (load * 1000 * X) / (48 * Ec * Ie_i) * (3 * L ** 2 - 4 * X ** 2)
            if (delta_i < 0) or (math.isnan(delta_i)) or (math.isinf(delta_i)):
                delta_i = 0
        else:
            delta_i = 0
        smooth_deflections.append(delta_i)
    smooth_deflections[0] = 0

    main_percentages = np.arange(0, 110, 10)
    main_loads = Pa * (main_percentages / 100)
    main_deflections = []
    for load in main_loads:
        Ma_i = (load * 1000) / 2 * X
        smoothing_range = 0.1 * Mcr if Mcr != 0 else 1
        x = (Ma_i - Mcr) / smoothing_range if smoothing_range != 0 else 0
        w = 0.5 * (1 + np.tanh(x))
        if Ma_i < 0.01:
            Ie_i = Ig
        else:
            ratio_i = (Mcr / Ma_i) ** 3 if Ma_i > 0 else 1
            Ie_uncracked = Ig
            Ie_cracked = beta_d * ratio_i * Ig + lambda_e * (1 - ratio_i) * Icr
            Ie_i = (1 - w) * Ie_uncracked + w * Ie_cracked
            Ie_i = min(Ie_i, Ig)
        if Ec != 0 and Ie_i != 0:
            delta_i = (load * 1000 * X) / (48 * Ec * Ie_i) * (3 * L ** 2 - 4 * X ** 2)
            if (delta_i < 0) or (math.isnan(delta_i)) or (math.isinf(delta_i)):
                delta_i = 0
        else:
            delta_i = 0
        main_deflections.append(delta_i)
    main_deflections[0] = 0

    df = pd.DataFrame({
        "Load (kN)": [round(l, 2) for l in Pa * (np.arange(0, 101, 5) / 100)],
        "Deflection (mm)": [round(d, 5) for d in np.interp(Pa * (np.arange(0, 101, 5) / 100), smooth_loads, smooth_deflections)]
    })

    st.markdown("#### Load-Deflection Curve")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Load-Deflection Data')
    output.seek(0)
    st.download_button(
        label="Download Table as Excel",
        data=output,
        file_name='load_deflection_data.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('white')
    ax.plot(smooth_deflections, smooth_loads, linestyle="-", marker=".", markersize=10, linewidth=2.5, color="red")
    ax.plot(main_deflections, main_loads, marker="p", markersize=10, linestyle="None", color="red")
    for defl, load, pct in zip(main_deflections, main_loads, main_percentages):
        if pct >= 20:
            ax.hlines(load, 0, defl, linestyles='dotted', linewidth=1)
            ax.vlines(defl, 0, load, linestyles='dotted', linewidth=1)
            ax.text(defl + 0.7, load, f"{int(pct)}%", va='center', fontsize=11, color="blue")
    ax.set_xlabel("Deflection (mm)")
    ax.set_ylabel("Load (kN)")
    ax.set_title("Load-Deflection Curve")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend().remove()
    st.pyplot(fig)

# ===== Step-by-step Log (unchanged with slight tweaks) =====
if 'show_log' not in st.session_state:
    st.session_state.show_log = False

col_log1, col_log2 = st.columns([1,1])
with col_log1:
    if st.button("Show step-by-step calculations (Log)"):
        st.session_state.show_log = True
with col_log2:
    if st.button("Clear Log"):
        st.session_state.show_log = False

if st.session_state.show_log:
    st.subheader("Step-by-step Calculation Log")
    # Recompute deterministic results for the log
    logs = []
    fc = float(fc); Ef = float(Ef)  # ensure numbers
    Ec = 4700 * (fc ** 0.5)
    nf = Ef / Ec if Ec != 0 else 0
    if section_type == "T-section":
        A1 = b * y1; A2 = B * y2; At = A1 + A2 + (nf - 1) * Af
        yt = (A1 * (y1 / 2) + A2 * (y1 + y2 / 2) + (nf - 1) * Af * d_prime) / At if At != 0 else 0
        Ig = (b * y1 ** 3) / 12 + A1 * (y1 / 2 - yt) ** 2 + (B * y2 ** 3) / 12 + A2 * ((y1 + y2 / 2) - yt) ** 2 + ((nf - 1) * Af) * (yt - d_prime) ** 2
    else:
        A1 = b * t; A2 = 0; At = b * t + (nf - 1) * Af
        yt = ((b * t ** 2) / 2 + (nf - 1) * Af * d_prime) / At if At != 0 else 0
        Ig = (b * t ** 3) / 12 + b * t * (t / 2 - yt) ** 2 + ((nf - 1) * Af) * (yt - d_prime) ** 2

    beta_d = min(0.06 * (rho_f / rho_fb), 0.50) if rho_fb != 0 else 0.50
    Xd_ratio = X / d if d != 0 else 0
    if Xd_ratio <= 2.8:
        lambda_e = min(max(0.085 * Xd_ratio + 0.00813 * d_prime + 0.44, 0.80), 0.95)
    else:
        lambda_e = min(max(-0.0074 * Xd_ratio + 0.0094 * d_prime + 0.30, 0.55), 0.95)
    Ma = (Pa * 1000) / 2 * X
    fr = 0.62 * math.sqrt(fc)
    Mcr = fr * Ig / yt if yt != 0 else 0

    if section_type == "T-section":
        a = B / 2
    else:
        a = b / 2
    b_quad = nf * Af
    c = - (nf * Af) * d
    disc = b_quad ** 2 - 4 * a * c
    Z = 0 if (disc < 0 or a == 0) else (-b_quad + math.sqrt(disc)) / (2 * a)
    Icr = (B * Z ** 3 / 3 + (nf * Af) * (d - Z) ** 2) if section_type == "T-section" else (b * Z ** 3 / 3 + (nf * Af) * (d - Z) ** 2)

    if Ma < Mcr:
        Ie = Ig
    else:
        ratio = (Mcr / Ma) ** 3 if Ma > 0 else 0
        Ie = beta_d * ratio * Ig + (min(max(lambda_e, 0.0), 0.95)) * (1 - ratio) * Icr
        Ie = min(Ie, Ig)
    delta_max = (Pa * 1000 * X) / (48 * Ec * Ie) * (3 * L ** 2 - 4 * X ** 2) if (Ec != 0 and Ie != 0) else 0

    logs.append(f"1️⃣ Ec = 4700 * sqrt({fc}) = {Ec:.2f} MPa")
    logs.append(f"2️⃣ nf = Ef / Ec = {Ef} / {Ec:.2f} = {nf:.3f}")
    if section_type == "T-section":
        logs.append(f"3️⃣ Web area A1 = {b} * {y1} = {A1:.2f} mm²")
        logs.append(f"4️⃣ Flange area A2 = {B} * {y2} = {A2:.2f} mm²")
        logs.append(f"5️⃣ At = {A1:.2f} + {A2:.2f} + ({nf:.3f} - 1) * {Af} = {At:.2f} mm²")
        logs.append(f"6️⃣ yt = {yt:.2f} mm")
        logs.append(f"7️⃣ Ig = {Ig:.2f} mm⁴")
    else:
        logs.append(f"3️⃣ Area = {b} * {t} = {A1:.2f} mm²")
        logs.append(f"4️⃣ At = {At:.2f} mm²")
        logs.append(f"5️⃣ yt = {yt:.2f} mm")
        logs.append(f"6️⃣ Ig = {Ig:.2f} mm⁴")

    logs.append(f"8️⃣ βd = min(0.06*({rho_f}/{rho_fb}), 0.50) = {beta_d:.3f}")
    logs.append(f"9️⃣ λe = {lambda_e:.3f}")
    logs.append(f"🔟 Ma = (({Pa} × 1000) / 2) × {X} = {Ma:.2f} N·mm")
    logs.append(f"11️⃣ fr = 0.62 * sqrt({fc}) = {fr:.3f} MPa")
    logs.append(f"12️⃣ Mcr = fr * Ig / yt = {fr:.3f} * {Ig:.2f} / {yt:.2f} = {Mcr:.2f} N·mm")
    if Ma < Mcr:
        logs.append(f"❗️ Ma < Mcr ⇒ Uncracked ⇒ Ie = Ig = {Ig:.2f} mm⁴")
    else:
        logs.append(f"13️⃣ (Mcr/Ma)^3 = ({Mcr:.2f}/{Ma:.2f})^3 = {(Mcr/Ma)**3:.3f}")
        logs.append(f"14️⃣ Z = {Z:.3f} mm")
        logs.append(f"15️⃣ Icr = {Icr:.2f} mm⁴")
        logs.append(f"16️⃣ Ie = {Ie:.2f} mm⁴")
        logs.append(f"17️⃣ δ_max = {delta_max:.5f} mm")

    for line in logs:
        st.write(line)
    step_text = "\n".join(logs)
    st.download_button(
        label="Download Step-by-step Log (TXT)",
        data=step_text,
        file_name="step_by_step_log.txt",
        mime="text/plain"
    )
