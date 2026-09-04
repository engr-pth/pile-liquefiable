import streamlit as st
import numpy as np
import pandas as pd

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
    
    with col2:
        P_axial = st.number_input("Axial Load (MN)", value=9.4, step=0.1)
        d_cone = st.number_input("Cone Diameter (mm)", value=25.4, step=0.1)
        
    with col3:
        delta_cv = st.number_input("Interface Friction Angle (°)", value=20.0, step=1.0)

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
# Main Tabs (Example 1 & Example 2)
# ---------------------------------------------------------
tab_ex1, tab_ex2 = st.tabs(["📌 Example 1: Static Loading", "📊 Example 2: CPT Methods (MTD & Randolph)"])

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
    # Integration for Layer 1 (0-8m) & Layer 2 (8-20m)
    Q_s_ex1 = np.pi * D_0 * ((0.5 * 1.27 * (8**2)) + (0.5 * 3.28 * (20**2 - 8**2)))
    
    Q_u_ex1 = Q_b_ex1 + Q_s_ex1
    N_required_ex1 = (P_axial * 1000) / Q_u_ex1

    # Results Display using Metric Cards
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    res_col1.metric("Base Capacity ($Q_b$)", f"{Q_b_ex1:.0f} kN")
    res_col2.metric("Shaft Capacity ($Q_s$)", f"{Q_s_ex1:.0f} kN")
    res_col3.metric("Total Capacity ($Q_u$)", f"{Q_u_ex1:.0f} kN")
    res_col4.metric("Req. Piles (FOS=1)", f"{N_required_ex1:.2f}")

    st.info(f"💡 **Design Recommendation:** Select **5 piles** to satisfy FOS ≥ 2 (or **7 piles** for FOS ≥ 3).")

# =========================================================
# EXAMPLE 2
# =========================================================
with tab_ex2:
    st.subheader("6.2.2 Design based on CPT Data")
    
    # Calculations for CPT End Bearing
    qc_tip_book = 26.7 # Book's value for average qc near tip
    qb_qc_ratio = max(1 - 0.5 * np.log10((D_0 * 1000) / d_cone), 0.13)
    q_b_cpt = qb_qc_ratio * qc_tip_book
    Q_b_ex2 = q_b_cpt * ((np.pi / 4) * (D_0 ** 2)) * 1000 # kN

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

    # Sub-tabs for CPT Methods
    subtab1, subtab2 = st.tabs(["1️⃣ MTD Method (Jardine & Chow)", "2️⃣ Randolph et al. Method"])

    with subtab1:
        st.markdown("#### 6.2.2.1 & 6.2.2.2 MTD Method Results")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("End Bearing ($Q_b$)", f"{Q_b_ex2:.0f} kN")
        col_b.metric("Shaft Friction ($Q_s$)", f"{Q_s_mtd:.0f} kN")
        col_c.metric("Total Capacity ($Q_u$)", f"{Q_u_mtd:.0f} kN")
        col_d.metric("Req. Piles ($N$)", f"{N_piles_mtd:.1f} ≈ {np.ceil(N_piles_mtd*2):.0f} piles (FOS=2)")

        st.write("**Table 6.2 Calculation of shear stress variation with depth**")
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

        st.write("**Table 6.3 Calculation of shear stress using Randolph et al. method**")
        df_randolph = pd.DataFrame({
            "Depth z (m)": depths,
            "σ'v0 (kPa)": sigma_v0,
            "qc (MPa)": qc_values,
            "qc / σ'v0": [round((q*1000)/s, 2) if s!=0 else 0 for q, s in zip(qc_values, sigma_v0)],
            "Kmax": K_max_list,
            "Shear Stress τs (kPa)": tau_s_randolph
        })
        st.dataframe(df_randolph, height=350, use_container_width=True)
