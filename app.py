import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Pile Foundation Design - Examples 1 & 2", layout="wide")

st.title("🏗️ Pile Foundation Design in Liquefiable Soils")
st.caption("Based on Gopal Madabhushi & Jonathan Knappett Examples")

# NumPy version 2.0+ trapz compatibility handle
def custom_trapz(y, x):
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        return np.trapz(y, x)

# Tab selection for Example 1 and Example 2
example_tab = st.radio("Choose Example / သင်ခန်းစာ ရွေးချယ်ပါ:", ["📌 Example 1: Basic Pile Cap Capacity & Settlement", "📌 Example 2: CPT-Based Shaft Friction & End Bearing (MTD & Randolph)"], horizontal=True)

st.divider()

if "Example 1" in example_tab:
    st.header("Example 1: Basic Pile Group Capacity & Settlement Analysis")
    
    with st.expander("⚙️ **Input Parameters for Example 1**", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            P_total = st.number_input("Total Structural Load, P (kN)", value=5000.0, step=100.0)
            D_pile = st.number_input("Pile Diameter, D (m)", value=0.6, step=0.05)
            L_pile = st.number_input("Pile Length, L (m)", value=15.0, step=1.0)
        with col2:
            n_rows = st.number_input("Number of Rows (n_x)", value=3, step=1)
            n_cols = st.number_input("Number of Columns (n_y)", value=3, step=1)
            spacing = st.number_input("Pile Spacing, s (m)", value=1.8, step=0.1)
        with col3:
            cu = st.number_input("Undrained Shear Strength, $c_u$ (kPa)", value=50.0, step=5.0)
            alpha = st.number_input("Adhesion Factor, $\\alpha$", value=0.7, step=0.05)
            E_s = st.number_input("Soil Young's Modulus, $E_s$ (MPa)", value=25.0, step=5.0)

    # Calculations for Example 1
    N_piles = n_rows * n_cols
    A_base = (np.pi / 4) * (D_pile ** 2)
    Perimeter = np.pi * D_pile
    
    # Single pile capacity
    Q_b_single = 9 * cu * A_base # kN
    Q_s_single = alpha * cu * Perimeter * L_pile # kN
    Q_u_single = Q_b_single + Q_s_single
    
    # Group capacity (Block failure check)
    B_group = (n_cols - 1) * spacing + D_pile
    L_group = (n_rows - 1) * spacing + D_pile
    
    Q_b_group = 9 * cu * (B_group * L_group)
    Q_s_group = 2 * (B_group + L_group) * L_pile * cu
    Q_u_group_block = Q_b_group + Q_s_group
    
    Q_u_group_individual = N_piles * Q_u_single
    Q_u_final = min(Q_u_group_individual, Q_u_group_block)
    FOS = Q_u_final / P_total

    # Display Example 1 Results
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Piles", f"{N_piles} Nos ({n_rows}x{n_cols})")
    col_b.metric("Single Pile Capacity ($Q_u$)", f"{Q_u_single:.1f} kN")
    col_c.metric("Group Ultimate Capacity", f"{Q_u_final:.1f} kN")
    col_d.metric("Factor of Safety (FOS)", f"{FOS:.2f}", delta="Safe" if FOS>=2.0 else "Unsafe")

    st.subheader("📊 Calculation Summary")
    st.write(f"- **Individual Piles Total Capacity:** `{Q_u_group_individual:.1f} kN`")
    st.write(f"- **Block Failure Capacity:** `{Q_u_group_block:.1f} kN`")
    st.write(f"- **Design Dimensions:** Width $B = {B_group:.2f}\\text{{ m}}$, Length $L = {L_group:.2f}\\text{{ m}}$")

else:
    st.header("Example 2: CPT-Based Shaft Friction & End Bearing (MTD & Randolph Methods)")
    
    with st.expander("⚙️ **Input Parameters for Example 2**", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            D_0 = st.number_input("Pile Diameter, $D_0$ (m)", value=0.75, step=0.05)
            L_p = st.number_input("Pile Length, $L_p$ (m)", value=20.0, step=1.0)
        with col2:
            P_axial = st.number_input("Axial Load (MN)", value=9.4, step=0.1)
            d_cone = st.number_input("Cone Diameter (mm)", value=25.4, step=0.1)
        with col3:
            delta_cv = st.number_input("Interface Friction Angle (°)", value=20.0, step=1.0)

    # CPT Raw Data (Table 6.2 / 6.3)
    depths = np.arange(0, 21, 1)
    sigma_v0 = [0, 7, 14, 21, 28, 35, 42, 49, 56, 81, 90, 99, 108, 117, 126, 135, 144, 153, 162, 171, 180]
    qc_values = [0, 0.74, 1.19, 1.81, 1.82, 2.41, 2.79, 3.25, 3.43, 12.85, 13.77, 16.03, 17.94, 19.07, 17.88, 24.94, 20.93, 23.71, 22.97, 26.59, 28.89]

    # End Bearing Capacity
    qc_tip_book = 26.7 # Book's value for qc at 20m tip
    qb_qc_ratio = max(1 - 0.5 * np.log10((D_0 * 1000) / d_cone), 0.13)
    q_b = qb_qc_ratio * qc_tip_book
    A_b = (np.pi / 4) * (D_0 ** 2)
    Q_b = q_b * A_b * 1000 # kN

    # 1. MTD Method
    tau_s_mtd = [0.0]
    for z, sig_v, qc in zip(depths[1:], sigma_v0[1:], qc_values[1:]):
        tau = (qc / 45.0) * ((sig_v / 100.0) ** 0.13) * ((0.75 / z) ** 0.38) * np.tan(np.radians(delta_cv)) * 1000
        tau_s_mtd.append(round(tau, 2))

    int_tau_mtd = custom_trapz(tau_s_mtd, depths)
    Q_s_mtd = np.pi * D_0 * int_tau_mtd
    Q_u_mtd = Q_b + Q_s_mtd
    N_piles_mtd = (P_axial * 1000) / Q_u_mtd

    # 2. Randolph et al. Method
    K_max_list = [0.0]
    tau_s_randolph = [0.0]
    for z, sig_v, qc in zip(depths[1:], sigma_v0[1:], qc_values[1:]):
        qc_kpa = qc * 1000
        K_max = 0.015 * (qc_kpa / sig_v)
        K_max_list.append(round(K_max, 2))
        tau = (0.4 + (K_max - 0.4) * np.exp(-0.05 * z / D_0)) * sig_v * np.tan(np.radians(delta_cv))
        tau_s_randolph.append(round(tau, 2))

    int_tau_randolph = custom_trapz(tau_s_randolph, depths)
    Q_s_randolph = np.pi * D_0 * int_tau_randolph
    Q_u_randolph = Q_b + Q_s_randolph

    # Results Display
    tab1, tab2 = st.tabs(["1️⃣ MTD Method (Jardine & Chow)", "2️⃣ Randolph et al. Method"])

    with tab1:
        st.subheader("6.2.2.1 & 6.2.2.2 MTD Method Results")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("End Bearing ($Q_b$)", f"{Q_b:.0f} kN")
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

    with tab2:
        st.subheader("Randolph et al. (1994) Method Results")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("End Bearing ($Q_b$)", f"{Q_b:.0f} kN")
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
