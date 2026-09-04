import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# ---------------------------------------------------------
# Calculations
# ---------------------------------------------------------

# 1. End Bearing Capacity (Eq 6.9 - 6.11)
qc_20m = qc_values[-1] # 28.89 (at z=20m, but book uses 26.7 MPa for qc average around tip)
qc_tip_book = 26.7 # Book's value for qc at 20m tip

qb_qc_ratio = 1 - 0.5 * np.log10((D_0 * 1000) / d_cone)
qb_qc_ratio = max(qb_qc_ratio, 0.13)

q_b = qb_qc_ratio * qc_tip_book # MPa
A_b = (np.pi / 4) * (D_0 ** 2)
Q_b = q_b * A_b * 1000 # kN

# 2. MTD Method (Eq 6.12)
tau_s_mtd = []
for z, sig_v, qc in zip(depths, sigma_v0, qc_values):
    if z == 0:
        tau_s_mtd.append(0.0)
    else:
        # Eq 6.12: [ (qc / 45) * (sig_v' / 100)^0.13 * (0.75 / z)^0.38 ] * tan(20 deg)
        # qc in MPa, sig_v in kPa
        tau = (qc / 45.0) * ((sig_v / 100.0) ** 0.13) * ((0.75 / z) ** 0.38) * np.tan(np.radians(delta_cv)) * 1000
        tau_s_mtd.append(round(tau, 2))

# Integration for MTD (Eq 6.13)
int_tau_mtd = np.trapezoid(tau_s_mtd, depths)
Q_s_mtd = np.pi * D_0 * int_tau_mtd
Q_u_mtd = Q_b + Q_s_mtd
N_piles_mtd = (P_axial * 1000) / Q_u_mtd

# 3. Randolph et al. Method (Eq 6.16)
K_max_list = []
tau_s_randolph = []

for z, sig_v, qc in zip(depths, sigma_v0, qc_values):
    if z == 0:
        K_max_list.append(0.0)
        tau_s_randolph.append(0.0)
    else:
        # qc in MPa to kPa: qc * 1000
        qc_kpa = qc * 1000
        K_max = 0.015 * (qc_kpa / sig_v)
        K_max_list.append(round(K_max, 2))
        
        # Eq 6.16
        tau = (0.4 + (K_max - 0.4) * np.exp(-0.05 * z / D_0)) * sig_v * np.tan(np.radians(delta_cv))
        tau_s_randolph.append(round(tau, 2))

# Integration for Randolph (Eq 6.17)
int_tau_randolph = np.trapezoid(tau_s_randolph, depths)
Q_s_randolph = np.pi * D_0 * int_tau_randolph
Q_u_randolph = Q_b + Q_s_randolph

# ---------------------------------------------------------
# Display Results
# ---------------------------------------------------------

tab1, tab2 = st.tabs(["1️⃣ MTD Method (Jardine & Chow)", "2️⃣ Randolph et al. Method"])

with tab1:
    st.subheader("6.2.2.1 & 6.2.2.2 MTD Method Results")
    
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("End Bearing ($Q_b$)", f"{Q_b:.0f} kN")
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

with tab2:
    st.subheader("Randolph et al. (1994) Method Results")
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("End Bearing ($Q_b$)", f"{Q_b:.0f} kN")
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
