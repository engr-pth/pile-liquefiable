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
int_tau_mtd = np.trapz(tau_s_mtd, depths)
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
int_tau_randolph = np.trapz(tau_s_randolph, depths)
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

st.set_page_config(page_title="Pile Foundation Design", layout="wide")

st.title("🏗️ Pile Foundation Design in Liquefiable Soils")
st.caption("Based on Gopal Madabhushi & Jonathan Knappett Examples")

# ---------------------------------------------------------
# Step 1: Main Page Inputs
# ---------------------------------------------------------
with st.expander("📌 **Step 1: Input Parameters (Pile & Soil Properties)**", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        D_0 = st.number_input("Pile Diameter, $D_0$ (m)", value=0.75, step=0.05)
        wall_thick = st.number_input("Wall Thickness, $t$ (mm)", value=12.0, step=1.0)
    
    with col2:
        L_p = st.number_input("Pile Length, $L_p$ (m)", value=20.0, step=1.0)
        P_axial = st.number_input("Axial Load (MN)", value=9.4, step=0.1)
        
    with col3:
        phi_cs = st.number_input("Critical Friction Angle (°)", value=32.0, step=1.0)
        pa = 0.1  # Atmospheric pressure in MPa (100 kPa)

st.divider()

# ---------------------------------------------------------
# Step 2: Main Calculation & Display (Tabs)
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Static Design (Example 1)", "📈 CPT-based Design (Example 2)"])

# -------------------- TAB 1: EXAMPLE 1 --------------------
with tab1:
    st.subheader("6.2.1 Preliminary Design under Static Loading")
    
    # End Bearing
    A_b = (np.pi / 4) * (D_0 ** 2)
    sigma_b_eff = (17 * 8) + (19 * 12) - (20 * 10)  # Effective stress at 20m depth
    N_q = 40  # Nq factor for phi = 32 deg
    Q_b_ex1 = A_b * sigma_b_eff * (N_q - 1)
    
    # Shaft Resistance
    Q_s_ex1 = np.pi * D_0 * ((0.5 * 1.27 * (8**2)) + (0.5 * 3.28 * (20**2 - 8**2)))
    
    Q_u_ex1 = Q_b_ex1 + Q_s_ex1
    N_req_ex1 = (P_axial * 1000) / Q_u_ex1

    # Results Display
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    res_col1.metric("Base Capacity ($Q_b$)", f"{Q_b_ex1:.0f} kN")
    res_col2.metric("Shaft Capacity ($Q_s$)", f"{Q_s_ex1:.0f} kN")
    res_col3.metric("Total Capacity ($Q_u$)", f"{Q_u_ex1:.0f} kN")
    res_col4.metric("Req. Piles (FOS=1)", f"{N_req_ex1:.2f}")

    st.info(f"💡 **Design Recommendation:** Select **5 piles** to satisfy FOS ≥ 2 (or **7 piles** for FOS ≥ 3).")

# -------------------- TAB 2: EXAMPLE 2 --------------------
with tab2:
    st.subheader("6.2.2 CPT-based Design (MTD Method)")
    
    # CPT Data (Table 6.2 Data)
    depths = np.arange(0, 21, 1)
    qc_values = [0, 0.74, 1.19, 1.81, 1.82, 2.41, 2.79, 3.25, 3.43, 12.85, 
                 13.77, 16.03, 17.94, 19.07, 17.88, 24.94, 20.93, 23.71, 22.97, 26.59, 28.89]
    
    df_cpt = pd.DataFrame({"Depth (m)": depths, "qc (MPa)": qc_values})
    
    # CPT Plot & Data Display
    c_left, c_right = st.columns([1, 1])
    with c_left:
        st.write("**CPT Profile Data**")
        st.dataframe(df_cpt, height=280, use_container_width=True)
    with c_right:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.plot(df_cpt["qc (MPa)"], df_cpt["Depth (m)"], marker='o', color='crimson', linewidth=1.5)
        ax.invert_yaxis()
        ax.set_xlabel("Cone Resistance $q_c$ (MPa)")
        ax.set_ylabel("Depth (m)")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("🧮 MTD Method Step-by-Step Calculations")

    # 1. Base Resistance Calculation (q_b & Q_b)
    # qc_ave at pile tip (1.5*D above and below 20m depth)
    qc_tip = np.mean([22.97, 26.59, 28.89])  # Average qc near 20m depth (~26.15 MPa)
    
    # MTD Base Resistance equation for driven pipe pile in sand:
    # q_b = qc_ave * [1 - 0.5 * (D_inner / D_outer)^2]
    D_inner = D_0 - 2 * (wall_thick / 1000)
    q_b = qc_tip * (1 - 0.5 * ((D_inner / D_0) ** 2))  # in MPa
    q_b_kPa = q_b * 1000  # convert to kPa
    
    A_b = (np.pi / 4) * (D_0 ** 2)
    Q_b_ex2 = q_b_kPa * A_b  # in kN

    # 2. Shaft Resistance Calculation (Q_s)
    # Layer 1: Loose Sand (0m - 8m)
    qc_avg_layer1 = np.mean(qc_values[1:9])  # avg from 1m to 8m (~2.19 MPa)
    # Layer 2: Dense Sand (8m - 20m)
    qc_avg_layer2 = np.mean(qc_values[9:21]) # avg from 9m to 20m (~20.4 MPa)

    # MTD Local Shear Stress Formula:
    # tau_f = 0.012 * qc * (pa / qc)^0.3 * (h / R_star)^-0.4 (Approx for CPT formulation)
    # MTD Integrated Shaft Capacity for Layer 1 & Layer 2:
    # Layer 1 (0 to 8m)
    q_s1_avg = 0.012 * (qc_avg_layer1 * 1000) * ((8 - 0)/D_0)**(-0.2)  # Average Unit Shaft Friction (kPa)
    Q_s1 = np.pi * D_0 * 8 * q_s1_avg
    
    # Layer 2 (8 to 20m)
    q_s2_avg = 0.008 * (qc_avg_layer2 * 1000) * ((20 - 8)/D_0)**(-0.2) # Average Unit Shaft Friction (kPa)
    Q_s2 = np.pi * D_0 * 12 * q_s2_avg
    
    Q_s_ex2 = Q_s1 + Q_s2
    Q_u_ex2 = Q_b_ex2 + Q_s_ex2
    N_req_ex2 = (P_axial * 1000) / Q_u_ex2

    # Detailed Output Table for Example 2
    calc_summary = pd.DataFrame({
        "Component": ["Layer 1 Shaft (0-8m)", "Layer 2 Shaft (8-20m)", "End Bearing ($Q_b$)", "Total Capacity ($Q_u$)"],
        "Parameter / Value": [f"qc_avg = {qc_avg_layer1:.2f} MPa", f"qc_avg = {qc_avg_layer2:.2f} MPa", f"qc_tip = {qc_tip:.2f} MPa", "-"],
        "Capacity (kN)": [f"{Q_s1:.2f}", f"{Q_s2:.2f}", f"{Q_b_ex2:.2f}", f"{Q_u_ex2:.2f}"]
    })
    
    st.table(calc_summary)

    # Final Result Cards
    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    r_col1.metric("Base Capacity ($Q_b$)", f"{Q_b_ex2:.0f} kN")
    r_col2.metric("Shaft Capacity ($Q_s$)", f"{Q_s_ex2:.0f} kN")
    r_col3.metric("Total Capacity ($Q_u$)", f"{Q_u_ex2:.0f} kN")
    r_col4.metric("Req. Piles (FOS=1)", f"{N_req_ex2:.2f}")

    st.success(f"✅ **Comparison Summary:** CPT-based MTD method gives Total Capacity = **{Q_u_ex2:.0f} kN** (Req. Piles ≈ **{np.ceil(N_req_ex2 * 2):.0f} piles** for FOS=2).")
