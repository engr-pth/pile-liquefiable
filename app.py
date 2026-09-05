import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Pile Foundation Design WorkFlow", layout="wide")

st.title("🏗️ Geotechnical & Foundation Design Workflow")
st.caption("CPT-Based Analysis ➔ Static Empirical Capacity ➔ Dynamic Soil Stiffness ➔ SSI & Active Length")

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

# Dual Classification Engine: Soil Type (Clay/Sand) + qc Threshold
def classify_soil_behavior(soil_type, qc):
    if soil_type == "Clay":
        if qc < 1.5:
            return "NC Soft Clay", 1.10, 16.0
        else:
            return "OC Stiff Clay", 0.70, 18.0
    else:  # Sand
        if qc < 5.0:
            return "Loose / Silty Sand", 0.90, 17.0
        elif 5.0 <= qc < 15.0:
            return "Medium Dense Sand", 0.75, 18.5
        else:
            return "Dense Sand / Granular", 0.60, 20.0

# ---------------------------------------------------------
# Main Input Section
# ---------------------------------------------------------
with st.expander("⚙️ **Design Parameters & CPT Field Input (Click to Expand/Collapse)**", expanded=True):
    col_p1, col_p2, col_p3 = st.columns([1.1, 1.3, 1.2])
    
    with col_p1:
        st.subheader("1. Pile Properties")
        
        # 1. Pile Type Dropdown ပြင်ဆင်ခြင်း
        pile_type = st.selectbox(
            "Select Pile Type", 
            ["Closed-end Steel Pipe Pile", "Closed-end Spun Concrete Pile"]
        )
        
        D_0 = st.number_input("Pile Outer Diameter, $D_0$ (m)", value=0.75, step=0.05)
        
        # 2. Pile Type ပေါ်မူတည်၍ Elastic Modulus & Parameters ယူခြင်း
        if "Steel" in pile_type:
            t_wall = st.number_input("Wall Thickness, $t$ (m)", value=0.012, step=0.001, format="%.3f")
            E_pile = 210.0  # GPa (Steel)
            delta_cv = 20.0 # Steel-Soil Interface Friction Angle
            st.caption("ℹ️ **Steel Pipe:** $E_p = 210$ GPa, $\\delta_{cv} = 20^\circ$")
        else:
            # Closed-end Spun Concrete Pile
            t_wall = st.number_input("Wall Thickness, $t$ (m)", value=0.090, step=0.005, format="%.3f") # e.g. 90mm wall for Spun Pile
            fc_prime = st.number_input("Concrete Strength, $f'_c$ (MPa)", value=60.0, step=5.0) # High-Strength PHC Spun Pile
            E_pile = (4700 * np.sqrt(fc_prime)) / 1000.0  # GPa
            delta_cv = 26.25 # Concrete-Soil Interface Friction Angle
            st.caption(f"ℹ️ **Spun Concrete:** $E_p = {E_pile:.2f}$ GPa, $\\delta_{{cv}} = 26.25^\circ$")

        L_p = st.number_input("Pile Length, $L_p$ (m)", value=20.0, step=1.0)
        P_axial = st.number_input("Axial Load (MN)", value=9.4, step=0.1)

    with col_p2:
        st.subheader("2. CPT Field Soil Test Input")
        st.caption("`Soil Type` (Clay/Sand) နှင့် CPT $q_c$ (MPa) တန်ဖိုးများကို Depth အလိုက် ရွေးချယ်ရိုက်ထည့်ပါ။")
        
        # Default CPT Profile Data Frame
        default_cpt_df = pd.DataFrame({
            "Depth (m)": np.arange(0, 21, 1),
            "Soil Type": ["Clay", "Clay", "Clay"] + ["Sand"] * 18,
            "qc (MPa)": [0, 0.74, 1.19, 1.81, 1.82, 2.41, 2.79, 3.25, 3.43, 12.85, 13.77, 16.03, 17.94, 19.07, 17.88, 24.94, 20.93, 23.71, 22.97, 26.59, 28.89]
        })
        
        edited_cpt_df = st.data_editor(
            default_cpt_df,
            column_config={
                "Soil Type": st.column_config.SelectboxColumn(
                    "Soil Type",
                    options=["Clay", "Sand"],
                    required=True
                )
            },
            height=250,
            num_rows="fixed",
            use_container_width=True
        )
        
        depths = edited_cpt_df["Depth (m)"].values
        user_soil_types = edited_cpt_df["Soil Type"].values
        qc_values = edited_cpt_df["qc (MPa)"].values

    with col_p3:
        st.subheader("3. Dynamic Correlation Settings")
        
        classified_descriptions = []
        e_values = []
        gamma_values = []
        
        for stype, q in zip(user_soil_types, qc_values):
            desc, e_val, g_val = classify_soil_behavior(stype, q)
            classified_descriptions.append(desc)
            e_values.append(e_val)
            gamma_values.append(g_val)

        # Dynamic Sigma_v0 calculation based on Effective Unit Weight
        sigma_v0 = [0.0]
        for i in range(1, len(depths)):
            dz = depths[i] - depths[i-1]
            gamma_eff = max(gamma_values[i] - 9.81, 7.0)
            sigma_v0.append(sigma_v0[-1] + gamma_eff * dz)

        e_silty = e_values[4] if len(e_values) > 4 else 0.90

        # ---------------------------------------------------------
        # AUTO-DETLECTION & CLASSIFICATION METHOD SELECTION
        # ---------------------------------------------------------
        method_choice = st.radio(
            "Select Soil Profile Classification Method:",
            [
                "Method 1: Dominant Soil Type in Active Depth (10D₀)",
                "Method 2: Weighted Average Exponent (n_eq) in Active Depth (10D₀)"
            ]
        )

        active_depth_limit = 10 * D_0  # Active Zone Depth Limit (e.g., 10 * 0.75 = 7.5m)
        active_mask = depths <= active_depth_limit
        active_descriptions = np.array(classified_descriptions)[active_mask]

        if "Method 1" in method_choice:
            # Method 1 Logic: Dominant Soil Type
            sand_count = sum(1 for d in active_descriptions if "Sand" in d)
            soft_clay_count = sum(1 for d in active_descriptions if "NC" in d or "Soft" in d)
            stiff_clay_count = sum(1 for d in active_descriptions if "OC" in d or "Stiff" in d)

            if sand_count >= max(soft_clay_count, stiff_clay_count):
                auto_profile = "Parabolic (Sand / Silty Sand)"
                exponent = 0.22
            elif soft_clay_count >= stiff_clay_count:
                auto_profile = "Linear (Soft Clay)"
                exponent = 0.20
            else:
                auto_profile = "Constant (OC Clay)"
                exponent = 0.25
        else:
            # Method 2 Logic: Weighted Average Exponent
            exponents = []
            for desc in classified_descriptions:
                if "Sand" in desc:
                    exponents.append(0.22)
                elif "NC" in desc or "Soft" in desc:
                    exponents.append(0.20)
                else:
                    exponents.append(0.25)

            active_exponents = np.array(exponents)[active_mask]
            exponent = round(float(np.mean(active_exponents)), 3)
            auto_profile = f"Interbedded Layer (Weighted Average n = {exponent})"

        st.text_input("Auto-Detected Soil Profile Type", value=auto_profile, disabled=True)
        st.info(f"💡 Active Depth Limit ($10D_0$): **{active_depth_limit:.2f} m** | Assigned Exponent: **n = {exponent}**")

        a_g = st.number_input("PGA, $a_g$ (g)", value=0.2, step=0.05)
        M_w = st.number_input("Earthquake Magnitude, $M_w$", value=6.0, step=0.1)
        apply_msf = st.checkbox("Apply MSF to $\\tau_{max}$?", value=False)

# ---------------------------------------------------------
# Dynamic Calculations
# ---------------------------------------------------------
K_0 = 0.46
sigma_v0_4m = sigma_v0[4] if len(sigma_v0) > 4 else 28.0
p_prime = ((1 + 2 * K_0) / 3) * sigma_v0_4m
G_0 = 100 * (((3 - e_silty) ** 2) / (1 + e_silty)) * np.sqrt(p_prime / 1000.0)

z_mid = 4.0
r_d = 1 - 0.01 * z_mid
tau_max_raw = 0.65 * a_g * 68.0 * r_d

if apply_msf:
    MSF = 6.9 * np.exp(-M_w / 4.0) - 0.058
    tau_max = tau_max_raw / MSF
else:
    MSF = 1.0
    tau_max = tau_max_raw

gamma_r, c_exponent = 2e-4, 0.79
gamma_sol = solve_shear_strain(tau_max, G_0 * 1000, gamma_r, c_exponent)
G_ratio = 1 / ((1 + (gamma_sol / gamma_r)) ** c_exponent)

# ---------------------------------------------------------
# Tabs Section
# ---------------------------------------------------------
tab_ex2, tab_ex1, tab_ex3, tab_ex4 = st.tabs([
    "📊 Step 1: CPT-Based Capacity (Ex 2)", 
    "📌 Step 2: Broms Static Capacity (Ex 1)", 
    "🌊 Step 3: Dynamic Soil Stiffness (Ex 3)",
    "📏 Step 4: Active Length & SSI (Ex 4)"
])

# =========================================================
# STEP 1: CPT METHOD (Jardine & Chow 1996 / ICP Method for Closed-end Driven Piles)
# =========================================================
with tab_ex2:
    st.subheader(f"6.2.2 CPT-Based Axial Capacity ({pile_type})")
    
    d_cone = 25.4  # Cone equivalent diameter (mm)
    qc_tip_book = qc_values[-1]
    
    # ICP Method End Bearing Factor for Closed-End Displacement Piles
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

    st.markdown("---")
    st.subheader("📋 Dynamic Soil Classification Table (Calculated from CPT Data)")
    
    summary_df = pd.DataFrame({
        "Depth (m)": depths,
        "Selected Soil Behavior": user_soil_types,
        "qc (MPa)": qc_values,
        "Auto-Calculated σ'v0 (kPa)": np.round(sigma_v0, 2),
        "Classified Soil Type & Density": classified_descriptions,
        "Mapped Void Ratio (e)": e_values
    })
    st.dataframe(summary_df, use_container_width=True)

# =========================================================
# STEP 2: BROMS STATIC METHOD (DYNAMIC FORMULA)
# =========================================================
with tab_ex1:
    st.subheader("6.2.1 Preliminary Design under Static Loading (Broms 1966)")
    
    H_top = 8.0  # Layer 1 depth (m)
    A_b = (np.pi / 4) * (D_0 ** 2)
    
    # Effective stress at pile tip
    sigma_b_eff = sigma_v0[-1]
    N_q = 40  
    Q_b_ex1 = A_b * sigma_b_eff * (N_q - 1)

    # Pile Material Parameters (Broms 1966, Table 1.1)
    if "Steel" in pile_type:
        K_s1, K_s2 = 0.5, 1.0
        delta_cv = 20.0
        st.info("ℹ️ **Broms (1966) Steel Parameters:** $\\delta_{cv} = 20^\\circ$, $K_{s1}=0.5$, $K_{s2}=1.0$")
    else:
        K_s1, K_s2 = 1.0, 2.0
        # For concrete, delta_cv = 0.75 * phi' (assuming phi' = 35° -> delta_cv = 26.25°)
        delta_cv = 26.25
        st.info("ℹ️ **Broms (1966) Concrete Parameters:** $\\delta_{cv} = 26.25^\\circ$, $K_{s1}=1.0$, $K_{s2}=2.0$")

    # Effective Unit Weights (kN/m³)
    gamma_sub1 = 17.0 - 10.0  # Layer 1 (0 to 8m)
    gamma_sub2 = 19.0 - 10.0  # Layer 2 (8 to 20m)

    tan_delta = np.tan(np.radians(delta_cv))

    # Integrations of Effective Vertical Stress (∫ σ'v dz)
    int_sigma_v1 = 0.5 * gamma_sub1 * (H_top ** 2)
    int_sigma_v2 = 0.5 * gamma_sub2 * (L_p ** 2 - H_top ** 2)

    # Unit Shaft Resistance Integrals (K_s * tan(delta) * ∫ σ'v dz)
    Q_s_layer1 = K_s1 * tan_delta * int_sigma_v1
    Q_s_layer2 = K_s2 * tan_delta * int_sigma_v2

    # Total Shaft Capacity Q_s = Perimeter (π * D) * Total Unit Shaft Resistance
    Q_s_ex1 = np.pi * D_0 * (Q_s_layer1 + Q_s_layer2)
    
    # Total Ultimate Bearing Capacity Q_u
    Q_u_ex1 = Q_b_ex1 + Q_s_ex1
    N_required_ex1 = (P_axial * 1000) / Q_u_ex1

    # Display Results
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    res_col1.metric("Base Capacity ($Q_b$)", f"{Q_b_ex1:.0f} kN")
    res_col2.metric("Shaft Capacity ($Q_s$)", f"{Q_s_ex1:.0f} kN")
    res_col3.metric("Total Capacity ($Q_u$)", f"{Q_u_ex1:.0f} kN")
    res_col4.metric("Req. Piles (FOS=1)", f"{N_required_ex1:.2f}")
# =========================================================
# STEP 3: DYNAMIC SOIL STIFFNESS
# =========================================================
with tab_ex3:
    st.subheader("6.3.1 Soil Stiffness and Natural Frequency")
    
    gamma_percent = gamma_sol * 100
    G_s = G_ratio * G_0
    nu = 0.5
    E_s = 2 * G_s * (1 + nu)
    v_s = np.sqrt((G_s * 1e6) / 1700.0)
    f_n = v_s / (4 * H_top)

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
# STEP 4: ACTIVE LENGTH & FLEXIBILITY
# =========================================================
with tab_ex4:
    st.subheader(f"6.3.2 Effective Active Length and Pile Flexibility ({pile_type})")

    # Hollow Section Moment of Inertia (I_p) Calculation
    D_i = D_0 - (2 * t_wall)
    I_p = (np.pi / 64) * (D_0**4 - D_i**4)
    
    if "Steel" in pile_type:
        E_p_corrected = E_pile / ((D_0**4) / (D_0**4 - D_i**4))
    else:
        E_p_corrected = E_pile

    sigma_v0_D0 = 7.0 * D_0  
    p_prime_D0 = ((1 + 2 * K_0) / 3) * sigma_v0_D0
    G_0_D0 = 100 * (((3 - e_silty) ** 2) / (1 + e_silty)) * np.sqrt(p_prime_D0 / 1000.0)
    G_s_D0 = G_ratio * G_0_D0  
    E_sD = 3 * G_s_D0          

    # Active length calculated using the dynamic exponent value based on chosen method
    L_ad = 2 * D_0 * ((E_p_corrected * 1e9) / (E_sD * 1e6)) ** exponent
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
    col_e2.metric(f"Soil Modulus ($E_{{sD}}$)", f"{E_sD:.2f} MPa")
    col_e3.metric("Effective Active Length ($L_{ad}$)", f"{L_ad:.2f} m")
    col_e4.metric("Flexural Rigidity ($E_p I_p$)", f"{E_I/1e6:.1f} MN·m²")

    col_e5, col_e6, col_e7, col_e8 = st.columns(4)
    col_e5.metric("Elastic Length ($T_u$, k=200)", f"{T_u:.3f} m")
    col_e6.metric("Relative Length ($Z_L$)", f"{Z_L:.2f} ({classify_behavior(Z_L)})")
    col_e7.metric("Elastic Length ($T_l$, k=2000)", f"{T_l:.3f} m")
    col_e8.metric("Relative Length ($Z_u$)", f"{Z_u:.2f} ({classify_behavior(Z_u)})")
