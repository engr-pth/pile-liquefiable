import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Pile Foundation Design", layout="wide")

st.title("🏗️ Pile Foundation Design in Liquefiable Soils")
st.caption("Supports Steel Pipe & Bored Concrete Piles with Dynamic MSF Option")

# ---------------------------------------------------------
# Step 1: Input Parameters
# ---------------------------------------------------------
with st.expander("📌 **Input Parameters**", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pile_type = st.selectbox("Select Pile Type", ["Steel Pipe Pile", "Bored Concrete Pile"])
        D_0 = st.number_input("Pile Diameter, $D_0$ (m)", value=0.75, step=0.05)
        L_p = st.number_input("Pile Length, $L_p$ (m)", value=20.0, step=1.0)
        e_silty = st.number_input("Void Ratio (Silty Sand), $e$", value=0.9, step=0.05)

    with col2:
        P_axial = st.number_input("Axial Load (MN)", value=9.4, step=0.1)
        d_cone = st.number_input("Cone Diameter (mm)", value=25.4, step=0.1)
        a_g = st.number_input("Peak Ground Acceleration, $a_g$ (g)", value=0.2, step=0.05)
        
        if pile_type == "Steel Pipe Pile":
            t_wall = st.number_input("Pile Wall Thickness, $t$ (m)", value=0.012, step=0.001)
            E_pile = st.number_input("Steel Modulus, $E_p$ (GPa)", value=210.0, step=5.0)
            default_delta = 20.0
        else:
            t_wall = D_0 / 2.0  # Solid Circular Section
            fc_prime = st.number_input("Concrete Strength, $f'_c$ (MPa)", value=30.0, step=5.0)
            E_pile = (4700 * np.sqrt(fc_prime)) / 1000.0  # ACI 318
            st.info(f"Calculated Concrete $E_c$: **{E_pile:.2f} GPa**")
            default_delta = 30.0

    with col3:
        delta_cv = st.number_input("Interface Friction Angle (°)", value=default_delta, step=1.0)
        rho_soil = st.number_input("Soil Density, $\\rho$ (kg/m³)", value=1700.0, step=50.0)
        H_layer = st.number_input("Silty Sand Layer Thickness, $H$ (m)", value=8.0, step=0.5)

    st.markdown("---")
    st.markdown("##### 🌊 Earthquake Magnitude & Scaling Options")
    col_msf1, col_msf2 = st.columns(2)
    with col_msf1:
        M_w = st.number_input("Earthquake Magnitude, $M_w$", value=6.0, step=0.1)
    with col_msf2:
        st.write("")
        st.write("")
        apply_msf = st.checkbox("Apply Magnitude Scaling Factor (MSF) to $\\tau_{max}$?", value=False)

st.divider()

# CPT Raw Data
depths = np.arange(0, 21, 1)
sigma_v0 = [0, 7, 14, 21, 28, 35, 42, 49, 56, 81, 90, 99, 108, 117, 126, 135, 144, 153, 162, 171, 180]
qc_values = [0, 0.74, 1.19, 1.81, 1.82, 2.41, 2.79, 3.25, 3.43, 12.85, 13.77, 16.03, 17.94, 19.07, 17.88, 24.94, 20.93, 23.71, 22.97, 26.59, 28.89]

def integrate_trapz(y, x):
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        return np.trapz(y, x)

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
# Dynamic Shear Stress Calculation (with optional MSF)
# ---------------------------------------------------------
K_0 = 0.46
sigma_v0_4m = 28.0
p_prime = ((1 + 2 * K_0) / 3) * sigma_v0_4m
G_0 = 100 * (((3 - e_silty) ** 2) / (1 + e_silty)) * np.sqrt(p_prime / 1000.0)

z_mid = 4.0
r_d = 1 - 0.01 * z_mid
tau_max_raw = 0.65 * a_g * 68.0 * r_d

if apply_msf:
    # Idriss MSF formula
    MSF = 6.9 * np.exp(-M_w / 4.0) - 0.058
    tau_max = tau_max_raw / MSF
else:
    MSF = 1.0
    tau_max = tau_max_raw

gamma_r, c_exponent = 2e-4, 0.79
gamma_sol = solve_shear_strain(tau_max, G_0 * 1000, gamma_r, c_exponent)
G_ratio = 1 / ((1 + (gamma_sol / gamma_r)) ** c_exponent)

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_ex1, tab_ex2, tab_ex3, tab_ex4 = st.tabs([
    "📌 Example 1: Static Loading", 
    "📊 Example 2: CPT Methods", 
    "🌊 Example 3: Dynamic Soil Stiffness",
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

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("End Bearing ($Q_b$)", f"{Q_b_ex2:.0f} kN")
    col_b.metric("Shaft Friction ($Q_s$)", f"{Q_s_mtd:.0f} kN")
    col_c.metric("Total Capacity ($Q_u$)", f"{Q_u_mtd:.0f} kN")
    col_d.metric("Req. Piles ($N$)", f"{N_piles_mtd:.1f}")

# =========================================================
# EXAMPLE 3
# =========================================================
with tab_ex3:
    st.subheader("6.3.1 Example 3: Soil Stiffness and Natural Frequency")

    if apply_msf:
        st.info(f"💡 **MSF Applied ($M_w={M_w:.1f}$):** MSF = {MSF:.2f} | Equiv. $\\tau_{{max}}$ = {tau_max:.2f} kPa")
    else:
        st.info(f"💡 **Direct Textbook Approach:** Raw $\\tau_{{max}}$ = {tau_max:.2f} kPa")

    gamma_percent = gamma_sol * 100
    G_s = G_ratio * G_0
    nu = 0.5
    E_s = 2 * G_s * (1 + nu)
    v_s = np.sqrt((G_s * 1e6) / rho_soil)
    f_n = v_s / (4 * H_layer)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confining Stress ($p'$)", f"{p_prime:.2f} kPa")
    m2.metric("Design Shear Stress ($\\tau_{max}$)", f"{tau_max:.2f} kPa")
    m3.metric("Initial Shear Modulus ($G_0$)", f"{G_0:.2f} MPa")
    m4.metric("Shear Strain ($\\gamma$)", f"{gamma_percent:.3f} %")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Modulus Ratio ($G_s / G_0$)", f"{G_ratio*100:.1f} %")
    m6.metric("Secant Modulus ($G_s$)", f"{G_s:.2f} MPa")
    m7.metric("Secant $E_s$", f"{E_s:.2f} MPa")
    m8.metric("Natural Freq. ($f_n$)", f"{f_n:.2f} Hz")

# =========================================================
# EXAMPLE 4
# =========================================================
with tab_ex4:
    st.subheader(f"6.3.2 Example 4: Effective Length and Flexibility ({pile_type})")

    if pile_type == "Steel Pipe Pile":
        D_i = D_0 - (2 * t_wall)
        E_p_corrected = E_pile / ((D_0**4) / (D_0**4 - D_i**4))
        I_p = (np.pi / 64) * (D_0**4 - D_i**4)
    else:
        D_i = 0.0
        E_p_corrected = E_pile
        I_p = (np.pi / 64) * (D_0**4)

    sigma_v0_D0 = 7.0 * D_0  
    p_prime_D0 = ((1 + 2 * K_0) / 3) * sigma_v0_D0
    
    p_prime_D0_MPa = p_prime_D0 / 1000.0
    G_0_D0 = 100 * (((3 - e_silty) ** 2) / (1 + e_silty)) * np.sqrt(p_prime_D0_MPa)
    
    G_s_D0 = G_ratio * G_0_D0  
    E_sD = 3 * G_s_D0          

    L_ad = 2 * D_0 * ((E_p_corrected * 1e9) / (E_sD * 1e6)) ** 0.22
    E_I = (E_pile * 1e9) * I_p

    k_lower, k_upper = 200.0, 2000.0

    T_u = ((E_I) / (k_lower * 1e3)) ** 0.2
    Z_L = L_p / T_u

    T_l = ((E_I) / (k_upper * 1e3)) ** 0.2
    Z_u = L_p / T_l

    def classify_behavior(z_val):
        if z_val > 5.0: return "Flexible"
        elif 2.5 <= z_val <= 5.0: return "Semi-Flexible"
        else: return "Rigid"

    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    col_e1.metric("Design Pile Modulus ($E_p$)", f"{E_p_corrected:.1f} GPa")
    col_e2.metric(f"Soil Modulus at {D_0:.2f}m ($E_{{sD}}$)", f"{E_sD:.2f} MPa")
    col_e3.metric("Effective Active Length ($L_{ad}$)", f"{L_ad:.2f} m")
    col_e4.metric("Flexural Rigidity ($E_p I_p$)", f"{E_I/1e6:.1f} MN·m²")

    col_e5, col_e6, col_e7, col_e8 = st.columns(4)
    col_e5.metric("Elastic Length ($T_u$, k=200)", f"{T_u:.3f} m")
    col_e6.metric("Relative Length ($Z_L$)", f"{Z_L:.2f} ({classify_behavior(Z_L)})")
    col_e7.metric("Elastic Length ($T_l$, k=2000)", f"{T_l:.3f} m")
    col_e8.metric("Relative Length ($Z_u$)", f"{Z_u:.2f} ({classify_behavior(Z_u)})")

    st.markdown("---")
    st.markdown("#### 📐 Example 4 Calculation Summary")

    ex4_summary_df = pd.DataFrame({
        "Parameter": [
            "Pile Type Selected",
            "MSF Option Status",
            "Effective Young's Modulus of Pile Section (E_p)",
            f"Effective Mean Confining Stress at Depth D₀={D_0:.2f}m (p')",
            f"Soil Young's Modulus at Depth D₀={D_0:.2f}m (E_sD)",
            "Effective Active Pile Length (L_ad)",
            "Elastic Length for k = 200 kN/m³ (T_u)",
            "Dimensionless Length Z_L (L / T_u)",
            "Elastic Length for k = 2000 kN/m³ (T_l)",
            "Dimensionless Length Z_u (L / T_l)"
        ],
        "Value": [
            pile_type,
            f"Applied (MSF={MSF:.2f})" if apply_msf else "Not Applied (Direct Textbook)",
            f"{E_p_corrected:.1f} GPa",
            f"{p_prime_D0:.2f} kPa",
            f"{E_sD:.2f} MPa",
            f"{L_ad:.2f} m",
            f"{T_u:.3f} m",
            f"{Z_L:.2f} ({classify_behavior(Z_L)})",
            f"{T_l:.3f} m",
            f"{Z_u:.2f} ({classify_behavior(Z_u)})"
        ],
        "Equation Reference": [
            "-", "-", "Eq. (6.33)", "Eq. (6.34)", "Eq. (6.37)", 
            "Eq. (6.38)", "Eq. (6.39)", "Eq. (6.40)", "Eq. (6.41)", "Eq. (6.42)"
        ]
    })
    st.table(ex4_summary_df)
