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
        e_silty = st.number_input("Void Ratio (Silty Sand), $e$", value=0.9, step=0.05)
        t_wall = st.number_input("Pile Wall Thickness, $t$ (m)", value=0.012, step=0.001)
    
    with col2:
        P_axial = st.number_input("Axial Load (MN)", value=9.4, step=0.1)
        d_cone = st.number_input("Cone Diameter (mm)", value=25.4, step=0.1)
        a_g = st.number_input("Peak Ground Acceleration, $a_g$ (g)", value=0.2, step=0.05)
        E_steel = st.number_input("Steel Young's Modulus, $E_p$ (GPa)", value=210.0, step=5.0)
        
    with col3:
        delta_cv = st.number_input("Interface Friction Angle (°)", value=20.0, step=1.0)
        rho_soil = st.number_input("Soil Density, $\\rho$ (kg/m³)", value=1700.0, step=50.0)
        H_layer = st.number_input("Silty Sand Layer Thickness, $H$ (m)", value=8.0, step=0.5)

st.divider()

# CPT Raw Data (Table 6.2 / 6.3)
depths = np.arange(0, 21, 1)
sigma_v0 = [0, 7, 14, 21, 28, 35, 42, 49, 56, 81, 90, 99, 108, 117, 126, 135, 144, 153, 162, 171, 180]
qc_values = [0, 0.74, 1.19, 1.81, 1.82, 2.41, 2.79, 3.25, 3.43, 12.85, 13.77, 16.03, 17.94, 19.07, 17.88, 24.94, 20.93, 23.71, 22.97, 26.59, 28.89]

# Helper function for trapezoidal integration
def integrate_trapz(y, x):
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        return np.trapz(y, x)

# Helper function for solving non-linear strain (Pure Python Binary Search)
def solve_shear_strain(tau_target, G0_kpa, gamma_r=2e-4, c=0.79):
    low, high = 1e-6, 0.1
    for _ in range(100):
        mid = (low + high) / 2.0
        tau_calc = G0_kpa * (mid / ((1 + (mid / gamma_r)) ** c))
        if tau_calc < tau_target:
            low = mid
        else:
            high = mid
    return mid

# ---------------------------------------------------------
# Main Tabs (Example 1, Example 2, Example 3, Example 4)
# ---------------------------------------------------------
tab_ex1, tab_ex2, tab_ex3, tab_ex4 = st.tabs([
    "📌 Example 1: Static Loading", 
    "📊 Example 2: CPT Methods (MTD & Randolph)", 
    "🌊 Example 3: Dynamic Soil Stiffness & $f_n$",
    "📏 Example 4: Effective Length & Flexibility"
])

# =========================================================
# EXAMPLE 1
# =========================================================
with tab_ex1:
    st.subheader("6.2.1 Preliminary Design under Static Loading")
    
    A_b = (np.pi / 4) * (D_0 ** 2)
    sigma_b_eff = (17 * 8) + (19 * 12) - (20 * 10)
    N_q = 40
    Q_b_ex1 = A_b * sigma_b_eff * (N_q - 1)
    
    Q_s_ex1 = np.pi * D_0 * ((0.5 * 1.27 * (8**2)) + (0.5 * 3.28 * (20**2 - 8**2)))
    Q_u_ex1 = Q_b_ex1 + Q_s_ex1
    N_required_ex1 = (P_axial * 1000) / Q_u_ex1

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

    K_0 = 0.46
    sigma_v0_4m = 28.0
    p_prime = ((1 + 2 * K_0) / 3) * sigma_v0_4m

    G_0 = 100 * (((3 - e_silty) ** 2) / (1 + e_silty)) * np.sqrt(p_prime / 1000.0)

    z_mid = 4.0
    r_d = 1 - 0.01 * z_mid
    sigma_v0_tot_4m = 68.0
    tau_max = 0.65 * a_g * sigma_v0_tot_4m * r_d

    gamma_r = 2e-4
    c_exponent = 0.79

    gamma_sol = solve_shear_strain(tau_max, G_0 * 1000, gamma_r, c_exponent)
    gamma_percent = gamma_sol * 100

    G_ratio = 1 / ((1 + (gamma_sol / gamma_r)) ** c_exponent)
    G_s = G_ratio * G_0

    nu = 0.5
    E_s = 2 * G_s * (1 + nu)

    v_s = np.sqrt((G_s * 1e6) / rho_soil)
    f_n = v_s / (4 * H_layer)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confining Stress ($p'$)", f"{p_prime:.2f} kPa")
    m2.metric("Max Shear Stress ($\\tau_{max}$)", f"{tau_max:.2f} kPa")
    m3.metric("Initial Shear Modulus ($G_0$)", f"{G_0:.2f} MPa")
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
            "Eq. (6.19)", "Eq. (6.20) / (2.2)", "Eq. (6.22)", "Eq. (6.26)", 
            "Eq. (6.28)", "Eq. (6.30)", "Eq. (6.31)", "Eq. (6.32)"
        ]
    })
    st.table(summary_df)

# =========================================================
# EXAMPLE 4
# =========================================================
with tab_ex4:
    st.subheader("6.3.2 Example 4: Effective Length and Flexibility of the Pile")

    # Dynamic calculation at depth z = D_0
    D_i = D_0 - (2 * t_wall)  # Inner diameter (m)
    
    # Corrected Young's Modulus for Hollow Steel Pile
    E_p_corrected = E_steel / ((D_0**4) / (D_0**4 - D_i**4))  # GPa

    # Overburden stress at depth z = D_0 m (γ'_silty = 7.0 kN/m³)
    sigma_v0_D0 = 7.0 * D_0  
    K_0 = 0.46
    p_prime_D0 = ((1 + 2 * K_0) / 3) * sigma_v0_D0  # Eq. 6.34
    
    # Eq. (2.2) / (6.35): p' in MPa unit
    p_prime_D0_MPa = p_prime_D0 / 1000.0
    G_0_D0 = 100 * (((3 - e_silty) ** 2) / (1 + e_silty)) * np.sqrt(p_prime_D0_MPa)  # Eq. 6.35
    
    # Degradation ratio from Example 3
    G_s_D0 = G_ratio * G_0_D0  # Eq. 6.36
    E_sD = 3 * G_s_D0          # Eq. 6.37

    # Effective active length
    L_ad = 2 * D_0 * ((E_p_corrected * 1e9) / (E_sD * 1e6)) ** 0.22  # m

    # Flexibility calculations
    I_p = (np.pi / 64) * (D_0**4 - D_i**4)
    E_I = (E_steel * 1e9) * I_p  # N.m²

    k_lower = 200.0   # kN/m³
    k_upper = 2000.0  # kN/m³

    T_u = ((E_I) / (k_lower * 1e3)) ** 0.2  # Eq. 6.39
    Z_L = L_p / T_u                         # Eq. 6.40

    T_l = ((E_I) / (k_upper * 1e3)) ** 0.2  # Eq. 6.41
    Z_u = L_p / T_l                         # Eq. 6.42

    def classify_behavior(z_val):
        if z_val > 5.0:
            return "Flexible"
        elif 2.5 <= z_val <= 5.0:
            return "Semi-Flexible"
        else:
            return "Rigid"

    # Metrics Display
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    col_e1.metric("Corrected Pile Modulus ($E_{p,corr}$)", f"{E_p_corrected:.1f} GPa")
    col_e2.metric(f"Soil Modulus at Depth {D_0:.2f}m ($E_{{sD}}$)", f"{E_sD:.2f} MPa")
    col_e3.metric("Effective Active Length ($L_{ad}$)", f"{L_ad:.2f} m")
    col_e4.metric("Pile Penetration in Sand", f"{max(0.0, L_ad - H_layer):.2f} m")

    col_e5, col_e6, col_e7, col_e8 = st.columns(4)
    col_e5.metric("Flexibility Elastic Length ($T_u$, k=200)", f"{T_u:.3f} m")
    col_e6.metric("Relative Length ($Z_L$)", f"{Z_L:.2f} ({classify_behavior(Z_L)})")
    col_e7.metric("Flexibility Elastic Length ($T_l$, k=2000)", f"{T_l:.3f} m")
    col_e8.metric("Relative Length ($Z_u$)", f"{Z_u:.2f} ({classify_behavior(Z_u)})")

    st.markdown("---")
    st.markdown("#### 📐 Example 4 Calculation Summary")

    # Dynamic Table Labels
    ex4_summary_df = pd.DataFrame({
        "Parameter": [
            "Corrected Young's Modulus of Hollow Steel Pile (E_p_corrected)",
            f"Effective Mean Confining Stress at Depth D₀={D_0:.2f}m (p')",
            f"Small-strain Shear Modulus at Depth D₀={D_0:.2f}m (G₀)",
            f"Degraded Shear Modulus at Depth D₀={D_0:.2f}m (G_s)",
            f"Soil Young's Modulus at Depth D₀={D_0:.2f}m (E_sD)",
            "Effective Active Pile Length (L_ad)",
            "Elastic Length for k = 200 kN/m³ (T_u)",
            "Dimensionless Length Z_L (L / T_u)",
            "Elastic Length for k = 2000 kN/m³ (T_l)",
            "Dimensionless Length Z_u (L / T_l)"
        ],
        "Value": [
            f"{E_p_corrected:.1f} GPa",
            f"{p_prime_D0:.2f} kPa",
            f"{G_0_D0:.2f} MPa",
            f"{G_s_D0:.2f} MPa",
            f"{E_sD:.2f} MPa",
            f"{L_ad:.2f} m",
            f"{T_u:.3f} m",
            f"{Z_L:.2f} ({classify_behavior(Z_L)})",
            f"{T_l:.3f} m",
            f"{Z_u:.2f} ({classify_behavior(Z_u)})"
        ],
        "Equation Reference": [
            "Eq. (6.33) / (2.47)", "Eq. (6.34)", "Eq. (6.35) / (2.2)", "Eq. (6.36)", 
            "Eq. (6.37)", "Eq. (6.38) / (2.9)", "Eq. (6.39) / (2.11)", 
            "Eq. (6.40) / (2.12)", "Eq. (6.41) / (2.11)", "Eq. (6.42) / (2.12)"
        ]
    })
    st.table(ex4_summary_df)

    st.info(f"📌 **Interpretation:** The true behavior of the pile lies between **{classify_behavior(Z_L)}** and **{classify_behavior(Z_u)}** depending on the actual soil modulus gradient $k$.")
