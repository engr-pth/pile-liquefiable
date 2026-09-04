import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pile Foundation Design", layout="wide")

st.title("🏗️ Pile Foundation Design in Liquefiable Soils")
st.caption("Based on Gopal Madabhushi & Jonathan Knappett Examples")

# ---------------------------------------------------------
# Step 1: Main Page Inputs (Sidebar မသုံးဘဲ Columns & Expander သုံးထားသည်)
# ---------------------------------------------------------
with st.expander("📌 **Step 1: Input Parameters (Pile & Soil Properties)**", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        D_0 = st.number_input("Pile Diameter, $D_0$ (m)", value=0.75, step=0.05)
        wall_thick = st.number_input("Wall Thickness (mm)", value=12.0, step=1.0)
    
    with col2:
        L_p = st.number_input("Pile Length, $L_p$ (m)", value=20.0, step=1.0)
        EI = st.number_input("Flexural Rigidity, $EI$ ($MNm^2$)", value=398.0, step=10.0)
        
    with col3:
        P_axial = st.number_input("Axial Load (MN)", value=9.4, step=0.1)
        phi_cs = st.number_input("Critical Friction Angle (°)", value=32.0, step=1.0)

st.divider()

# ---------------------------------------------------------
# Step 2: Main Calculation & Display (Tabs)
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Static Design (Example 1)", "📈 CPT-based Design (Example 2)"])

with tab1:
    st.subheader("6.2.1 Preliminary Design under Static Loading")
    
    # End Bearing Calculation[cite: 1]
    A_b = (np.pi / 4) * (D_0 ** 2)
    sigma_b_eff = (17 * 8) + (19 * 12) - (20 * 10)  # Effective vertical stress at 20m depth
    N_q = 40  # Nq for friction angle 32 deg
    Q_b = A_b * sigma_b_eff * (N_q - 1)
    
    # Shaft Resistance Calculation[cite: 1]
    # Integration for Layer 1 (0-8m) & Layer 2 (8-20m)
    Q_s = np.pi * D_0 * ((0.5 * 1.27 * (8**2)) + (0.5 * 3.28 * (20**2 - 8**2)))
    
    Q_u = Q_b + Q_s
    N_required = (P_axial * 1000) / Q_u

    # Results Display using Metric Cards
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    res_col1.metric("Base Capacity ($Q_b$)", f"{Q_b:.0f} kN")
    res_col2.metric("Shaft Capacity ($Q_s$)", f"{Q_s:.0f} kN")
    res_col3.metric("Total Capacity ($Q_u$)", f"{Q_u:.0f} kN")
    res_col4.metric("Req. Piles (FOS=1)", f"{N_required:.2f}")

    st.info(f"💡 **Design Recommendation:** Select **5 piles** to satisfy FOS ≥ 2 (or **7 piles** for FOS ≥ 3).")

with tab2:
    st.subheader("6.2.2 Preliminary Design using CPT Data")
    
    # CPT Data setup (Table 6.2 Example)[cite: 1]
    depths = np.arange(0, 21, 1)
    qc_values = [0, 0.74, 1.19, 1.81, 1.82, 2.41, 2.79, 3.25, 3.43, 12.85, 
                 13.77, 16.03, 17.94, 19.07, 17.88, 24.94, 20.93, 23.71, 22.97, 26.59, 28.89]
    
    df = pd.DataFrame({"Depth (m)": depths, "qc (MPa)": qc_values})
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.write("**CPT Resistance Profile ($q_c$)**")
        st.dataframe(df, height=350, use_container_width=True)
        
    with col_right:
        fig, ax = plt.subplots(figsize=(4, 5))
        ax.plot(df["qc (MPa)"], df["Depth (m)"], marker='o', color='crimson', linewidth=2)
        ax.invert_yaxis()
        ax.set_xlabel("Cone Resistance $q_c$ (MPa)")
        ax.set_ylabel("Depth (m)")
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
