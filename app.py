import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pile Foundation Design", layout="wide")

st.title("🏗️ Pile Foundation Design in Liquefiable Soils")
st.subheader("Based on Gopal Madabhushi & Jonathan Knappett Examples")

# Sidebar for Input Parameters
st.sidebar.header("Pile & Soil Properties")
D_0 = st.sidebar.number_input("Pile Diameter (m)", value=0.75, step=0.05)
L_p = st.sidebar.number_input("Pile Length (m)", value=20.0, step=1.0)
P_axial = st.sidebar.number_input("Axial Load (MN)", value=9.4, step=0.1)

# Tab Selection for Different Examples
tab1, tab2 = st.tabs(["Static Design (Example 1)", "CPT-based Design (Example 2)"])

with tab1:
    st.header("Static Design under Axial Loading")
    
    # 1. End Bearing
    A_b = (np.pi / 4) * (D_0 ** 2)
    sigma_b_eff = (17 * 8) + (19 * 12) - (20 * 10)  # Effective stress
    N_q = 40  # Nq factor for phi = 32 deg
    Q_b = A_b * sigma_b_eff * (N_q - 1)
    
    # 2. Shaft Resistance
    # Integration for two layers: 0-8m and 8-20m
    Q_s = np.pi * D_0 * ((0.5 * 1.27 * (8**2)) + (0.5 * 3.28 * (20**2 - 8**2)))
    
    Q_u = Q_b + Q_s
    N_required = (P_axial * 1000) / Q_u

    st.write(f"**Base Capacity ($Q_b$):** {Q_b:.2f} kN")
    st.write(f"**Shaft Resistance ($Q_s$):** {Q_s:.2f} kN")
    st.write(f"**Total Capacity ($Q_u$):** {Q_u:.2f} kN")
    st.info(f"**Required Piles (FOS = 1):** {N_required:.2f}")

with tab2:
    st.header("CPT-based Design (MTD Method)")
    # Generate CPT Data Table (Table 6.2 Example)
    depths = np.arange(0, 21, 1)
    qc_values = [0, 0.74, 1.19, 1.81, 1.82, 2.41, 2.79, 3.25, 3.43, 12.85, 
                 13.77, 16.03, 17.94, 19.07, 17.88, 24.94, 20.93, 23.71, 22.97, 26.59, 28.89]
    
    df = pd.DataFrame({"Depth (m)": depths, "qc (MPa)": qc_values})
    
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.plot(df["qc (MPa)"], df["Depth (m)"], marker='o', color='b')
    ax.invert_yaxis()
    ax.set_xlabel("qc (MPa)")
    ax.set_ylabel("Depth (m)")
    ax.grid(True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(df)
    with col2:
        st.pyplot(fig)
