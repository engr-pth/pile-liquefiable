import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

st.set_page_config(page_title="Pile Foundation Design", layout="wide")

st.title("🏗️ Pile Foundation Design in Liquefiable Soils")
st.caption("Based on Gopal Madabhushi & Jonathan Knappett Examples")

# ---------------------------------------------------------
# Step 1: Input Parameters
# ---------------------------------------------------------
with st.expander("📌 **Input Parameters**", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        D_0 = st.number_input("Pile Diameter, $D_0$ (m)", value=0.75, step=0.05)
        L_p = st.number_input("Pile Length, $L_p$ (m)", value=20.0, step=1.0)
        e_silty = st.number_input("Void Ratio (Silty Sand), $e$", value=0.9, step=0.05)
    
    with col2:
        P_axial = st.number_input("Axial Load (MN)", value=9.4, step=0.1)
        d_cone = st.number_input("Cone Diameter (mm)", value=25.4, step=0.1)
        a_g = st.number_input("Peak Ground Acceleration, $a_g$ (g)", value=0.2, step=0.05)
        
    with col3:
        delta_cv = st.number_input("Interface Friction Angle (°)", value=20.0, step=1.0)
        rho_soil = st.number_input("Soil Density, $\\rho$ (kg/m³)", value=1700.0, step=50.0)
        H_layer = st.number_input("Silty Sand Layer Thickness, $H$ (m)", value=8.0, step=0.5)

st.divider()

# CPT Raw Data (Table 6.2 / 6.3)
depths = np.arange(0, 21, 1)
sigma_v0 = [0, 7, 14, 21, 28, 35, 42, 49, 56, 81, 90, 99, 108, 117, 126, 135, 144, 153, 162, 171, 180]
qc_values = [0, 0.74, 1.19, 1.81, 1.82, 2.41, 2.79, 3.25, 3.43, 12.85, 13.77, 16.03, 17.94, 19.07, 17.88, 24.94, 20.93, 23.71, 22.97, 26.59, 28.89]

# Helper function for trapezoidal integration (compatible with numpy 2.0+)
def integrate_trapz(y, x):
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        return np.trapz(y, x)

# ---------------------------------------------------------
# Main Tabs (Example 1, Example 2, Example 3)
# ---------------------------------------------------------
tab_ex1, tab_ex2, tab_ex3 = st.tabs([
    "📌 Example 1: Static Loading", 
    "📊 Example 2: CPT Methods (MTD & Randolph)", 
    "🌊 Example 3: Dynamic Soil Stiffness & $f_n$"
])

# =========================================================
# EXAMPLE 1
# =========================================================
with tab_ex1:
    st.subheader("6.2.1 Preliminary Design under Static Loading")
    
    # End Bearing Calculation
    A_b = (np.pi / 4) * (D_0 ** 2)
    sigma_b_eff = (17 * 8) + (19 * 12) - (20 * 10)  # Effective vertical stress at 20m depth
    N_q = 40  # Nq for friction angle 32 deg
    Q_b_ex1 = A_b * sigma_b_eff * (N_q - 1)
    
    # Shaft Resistance Calculation
    Q_s_ex1 = np.pi * D_0 * ((0.5 * 1.27 * (8**2)) + (0.5 * 3.28 * (20**2 - 8**2)))
    
    Q_u_ex1 = Q_b_ex1 + Q_s_ex1
    N_required_ex1 = (P_axial * 1000) / Q_u_ex1

    # Results Display using Metric Cards
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    res_col1.metric("Base Capacity ($Q_b$)", f"{Q_b_ex1:.0f} kN")
    res_col2.metric("Shaft Capacity ($Q_s$)", f"{Q_s_ex1:.0f} kN")
    res_col3.metric("Total Capacity ($Q_u$)", f"{Q_u_ex1:.0f} kN")
    res_col4.metric("Req. Piles (FOS=1)", f"{N_required_ex1:.2f}")

    st.info("💡 **Design Recommendation:** Select **5 piles** to satisfy FOS ≥ 2 (or **7 piles** for FOS ≥ 3).")

# =========================================================
# EXAMPLE 2
# =========================================================
with tab_ex2:
    st.subheader("6.2.2 Design based on CPT Data")
    
    qc_tip_book = 26.7
    qb_qc_ratio = max(1 - 0.5 * np.log10((D_0 * 1000) / d_cone), 0.13)
    q_b_cpt = qb_qc_ratio * qc_tip_book
    Q_b_ex2 = q_b_cpt * ((np.pi / 4) * (D_0 ** 2)) * 1000

    # 1. MTD Method
    tau_s_mtd = []
    for z, sig_v, qc in zip(depths, sigma_v0, qc_values):
        if z == 0:
            tau_s_mtd.append(0.0)
        else:
            tau = (qc / 45.0) * ((sig_v / 100.0) ** 0.13) * ((0.75 / z) ** 0.38) * np.tan(np.radians(delta_cv)) * 1000
            tau_s_mtd.append(round(tau, 2))

    int_tau_mtd = integrate_trapz(tau_s_mtd, depths)
    Q_s_mtd = np.pi * D_0 * int_tau_mtd
    Q_u_mtd = Q_b_ex2 + Q_s_mtd
    N_piles_mtd = (P_axial * 1000) / Q_u_mtd

    # 2. Randolph et al. Method
    K_max_list = []
    tau_s_randolph = []
    for z, sig_v, qc in zip(depths, sigma_v0, qc_values):
        if z == 0:
            K_max_list.append(0.0)
            tau_s_randolph.append(0.0)
        else:
            qc_kpa = qc * 1000
            K_max = 0.015 * (qc_kpa / sig_v)
            K_max_list.append(round(K_max, 2))
            tau = (0.4 + (K_max - 0.4) * np.exp(-0.05 * z / D_0)) * sig_v * np.tan(np.radians(delta_cv))
            tau_s_randolph.append(round(tau, 2))

    int_tau_randolph = integrate_trapz(tau_s_randolph, depths)
    Q_s_randolph = np.pi * D_0 * int_tau_randolph
    Q_u_randolph = Q_b_ex2 + Q_s_randolph

    subtab1, subtab2 = st.tabs(["1️⃣ MTD Method (Jardine & Chow)", "2️⃣ Randolph et al. Method"])

    with subtab1:
        st.markdown("#### 6.2.2.1 & 6.2.2.2 MTD Method Results")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("End Bearing ($Q_b$)", f"{Q_b_ex2:.0f} kN")
        col_b.metric("Shaft Friction ($Q_s$)", f"{Q_s_mtd:.0f} kN")
        col_c.metric("Total Capacity ($Q_u$)", f"{Q_u_mtd:.0f} kN")
        col_d.metric("Req. Piles ($N$)", f"{N_piles_mtd:.1f} ≈ {np.ceil(N_piles_mtd*2):.0f} piles (FOS=2)")

        df_mtd = pd.DataFrame({
            "Depth z (m)": depths,
            "σ'v0 (kPa)": sigma_v0,
            "qc (MPa)": qc_values,
            "Shear Stress τs (kPa)": tau_s_mtd
        })
        st.dataframe(df_mtd, height=350, use_container_width=True)

    with subtab2:
        st.markdown("#### Randolph et al. (1994) Method Results")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("End Bearing ($Q_b$)", f"{Q_b_ex2:.0f} kN")
        col_b.metric("Shaft Friction ($Q_s$)", f"{Q_s_randolph:.0f} kN")
        col_c.metric("Total Capacity ($Q_u$)", f"{Q_u_randolph:.0f} kN")

        df_randolph = pd.DataFrame({
            "Depth z (m)": depths,
            "σ'v0 (kPa)": sigma_v0,
            "qc (MPa)": qc_values,
            "qc / σ'v0": [round((q*1000)/s, 2) if s!=0 else 0 for q, s in zip(qc_values, sigma_v0)],
            "Kmax": K_max_list,
            "Shear Stress τs (kPa)": tau_s_randolph
        })
        st.dataframe(df_randolph, height=350, use_container_width=True)

# =========================================================
# EXAMPLE 3
# =========================================================
with tab_ex3:
    st.subheader("6.3.1 Example 3: Soil Stiffness and Natural Frequency")

    # 1. Effective Stress & Small-Strain Shear Modulus (Eq 6.19 & 6.20)
    K_0 = 0.46
    sigma_v0_4m = 28.0  # kPa at z=4m (mid-depth of silty sand layer)
    p_prime = ((1 + 2 * K_0) / 3) * sigma_v0_4m  # Eq 6.19 (kPa)

    # G_0 calculation (Hardin & Drnevich, Eq 2.2 / 6.20)
    G_0 = 100 * (((3 - e_silty) ** 2) / (1 + e_silty)) * np.sqrt(p_prime / 1000)  # MPa

    # 2. Maximum Cyclic Shear Stress (Eq 6.21 & 6.22)
    z_mid = 4.0
    r_d = 1 - 0.01 * z_mid
    sigma_v0_tot_4m = 68.0  # total stress at z=4m (kPa)
    tau_max = 0.65 * a_g * sigma_v0_tot_4m * r_d  # kPa (Eq 6.22)

    # 3. Nonlinear Shear Strain Iteration (Eq 6.25 & 6.26)
    gamma_r = 2e-4  # Reference strain (0.02%)
    c_exponent = 0.79

    def eq_strain(gamma):
        # tau_max in kPa, G_0 in kPa
        return (G_0 * 1000) * (gamma / ((1 + (gamma / gamma_r)) ** c_exponent)) - tau_max

    gamma_sol = fsolve(eq_strain, 0.001)[0]
    gamma_percent = gamma_sol * 100

    # Degradation and Secant Shear Modulus (Eq 6.28)
    G_ratio = 1 / ((1 + (gamma_sol / gamma_r)) ** c_exponent)
    G_s = G_ratio * G_0  # MPa

    # Young's Modulus under undrained condition (nu = 0.5) (Eq 6.29 & 6.30)
    nu = 0.5
    E_s = 2 * G_s * (1 + nu)  # MPa

    # 4. Shear Wave Velocity & Natural Frequency (Eq 6.31 & 6.32)
    v_s = np.sqrt((G_s * 1e6) / rho_soil)  # m/s
    f_n = v_s / (4 * H_layer)  # Hz

    # Display Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confining Stress ($p'$)", f"{p_prime:.1f} kPa")
    m2.metric("Max Shear Stress ($\\tau_{max}$)", f"{tau_max:.2f} kPa")
    m3.metric("Initial Shear Modulus ($G_0$)", f"{G_0:.1f} MPa")
    m4.metric("Shear Strain ($\\gamma$)", f"{gamma_percent:.3f} %")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Modulus Ratio ($G_s / G_0$)", f"{G_ratio*100:.1f} %")
    m6.metric("Secant Modulus ($G_s$)", f"{G_s:.2f} MPa")
    m7.metric("Secant $E_s$", f"{E_s:.2f} MPa")
    m8.metric("Natural Freq. ($f_n$)", f"{f_n:.2f} Hz")

    st.markdown("---")
    st.markdown("#### 📐 Calculation Summary")
    
    summary_df = pd.DataFrame({
        "Parameter": [
            "Mean Effective Confining Stress (p')",
            "Small-strain Shear Modulus (G₀)",
            "Peak Cyclic Shear Stress (τ_max)",
            "Calculated Cyclic Shear Strain (γ)",
            "Degraded Secant Shear Modulus (G_s)",
            "Soil Young's Modulus (E_s)",
            "Shear Wave Velocity (v_s)",
            "Fundamental Natural Frequency (f_n)"
        ],
        "Value": [
            f"{p_prime:.2f} kPa",
            f"{G_0:.2f} MPa",
            f"{tau_max:.2f} kPa",
            f"{gamma_percent:.3f} % ({gamma_sol:.5f})",
            f"{G_s:.2f} MPa",
            f"{E_s:.2f} MPa",
            f"{v_s:.1f} m/s",
            f"{f_n:.2f} Hz"
        ],
        "Equation Reference": [
            "Eq. (6.19)", "Eq. (6.20)", "Eq. (6.22)", "Eq. (6.26)", 
            "Eq. (6.28)", "Eq. (6.30)", "Eq. (6.31)", "Eq. (6.32)"
        ]
    })
    st.table(summary_df)
