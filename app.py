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
        pile_type = st.selectbox(
            "Select Pile Type", 
            ["Closed-end Steel Pipe Pile", "Closed-end Spun Concrete Pile"]
        )
        
        D_0 = st.number_input("Pile Outer Diameter, $D_0$ (m)", value=0.75, step=0.05)
        
        if "Steel" in pile_type:
            t_wall = st.number_input("Wall Thickness, $t$ (m)", value=0.012, step=0.001, format="%.3f")
            E_pile = 210.0  # GPa (Steel)
            delta_cv = 20.0 # Steel-Soil Interface Friction Angle
            st.caption("ℹ️ **Steel Pipe:** $E_p = 210$ GPa, $\\delta_{cv} = 20^\circ$")
        else:
            t_wall = st.number_input("Wall Thickness, $t$ (m)", value=0.090, step=0.005, format="%.3f")
            fc_prime = st.number_input("Concrete Strength, $f'_c$ (MPa)", value=60.0, step=5.0)
            E_pile = (4700 * np.sqrt(fc_prime)) / 1000.0  # GPa
            delta_cv = 26.25 # Concrete-Soil Interface Friction Angle
            st.caption(f"ℹ️ **Spun Concrete:** $E_p = {E_pile:.2f}$ GPa, $\\delta_{{cv}} = 26.25^\circ$")

        L_p = st.number_input("Pile Length, $L_p$ (m)", value=20.0, step=1.0)
        P_axial = st.number_input("Axial Load (MN)", value=9.4, step=0.1)

    with col_p2:
        st.subheader("2. CPT Field Soil Test Input")
        st.caption("`Soil Type` နှင့် CPT $q_c$ (MPa) တန်ဖိုးများကို Depth အလိုက် ရိုက်ထည့်ပါ။")
        
        density_mode = st.radio(
            "Soil Unit Weight (γ) Selection Mode:",
            ["Auto (Correlate from CPT qc)", "Manual Input"],
            horizontal=True
        )

        default_cpt_df = pd.DataFrame({
            "Depth (m)": np.arange(0, 21, 1),
            "Soil Type": ["Clay", "Clay", "Clay"] + ["Sand"] * 18,
            "qc (MPa)": [0, 0.74, 1.19, 1.81, 1.82, 2.41, 2.79, 3.25, 3.43, 12.85, 13.77, 16.03, 17.94, 19.07, 17.88, 24.94, 20.93, 23.71, 22.97, 26.59, 28.89],
            "γ_manual (kN/m³)": [16.0, 16.0, 16.0] + [17.0] * 6 + [18.5] * 2 + [20.0] * 10
        })
        
        column_config = {
            "Soil Type": st.column_config.SelectboxColumn(
                "Soil Type",
                options=["Clay", "Sand"],
                required=True
            )
        }
        
        if density_mode == "Manual Input":
            column_config["γ_manual (kN/m³)"] = st.column_config.NumberColumn(
                "γ Total (kN/m³)",
                min_value=10.0,
                max_value=25.0,
                step=0.5,
                required=True
            )
        else:
            default_cpt_df = default_cpt_df.drop(columns=["γ_manual (kN/m³)"])

        edited_cpt_df = st.data_editor(
            default_cpt_df,
            column_config=column_config,
            height=250,
            num_rows="fixed",
            use_container_width=True
        )
        
        depths = edited_cpt_df["Depth (m)"].values
        user_soil_types = edited_cpt_df["Soil Type"].values
        qc_values = edited_cpt_df["qc (MPa)"].values
        if density_mode == "Manual Input":
            user_gamma_manual = edited_cpt_df["γ_manual (kN/m³)"].values

    with col_p3:
        st.subheader("3. Dynamic Correlation Settings")
        
        classified_descriptions = []
        e_values = []
        gamma_values = []
        
        for i, (stype, q) in enumerate(zip(user_soil_types, qc_values)):
            desc, e_val, g_val = classify_soil_behavior(stype, q)
            classified_descriptions.append(desc)
            e_values.append(e_val)
            
            if density_mode == "Manual Input":
                gamma_values.append(user_gamma_manual[i])
            else:
                gamma_values.append(g_val)

        # Dynamic Sigma_v0 calculation based on Effective Unit Weight
        sigma_v0 = [0.0]
        for i in range(1, len(depths)):
            dz = depths[i] - depths[i-1]
            gamma_eff = max(gamma_values[i] - 9.81, 7.0)
            sigma_v0.append(sigma_v0[-1] + gamma_eff * dz)

        e_silty = e_values[4] if len(e_values) > 4 else 0.90

        method_choice = st.radio(
            "Select Soil Profile Classification Method:",
            [
                "Method 1: Dominant Soil Type in Active Depth (10D₀)",
                "Method 2: Weighted Average Exponent (n_eq) in Active Depth (10D₀)"
            ]
        )

        active_depth_limit = 10 * D_0
        active_mask = depths <= active_depth_limit
        active_descriptions = np.array(classified_descriptions)[active_mask]

        if "Method 1" in method_choice:
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
# STEP 1: CPT METHOD (ICP / Jardine & Chow 1996)
# =========================================================
with tab_ex2:
    st.subheader(f"6.2.2 CPT-Based Axial Capacity ({pile_type})")
    
    d_cone = 25.4  # mm
    qc_tip_book = qc_values[-1]
    
    qb_qc_ratio = max(1 - 0.5 * np.log10((D_0 * 1000) / d_cone), 0.13)
    q_b_cpt = qb_qc_ratio * qc_tip_book
    A_b_cpt = (np.pi / 4) * (D_0 ** 2)
    Q_b_ex2 = q_b_cpt * A_b_cpt * 1000  # kN

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

    with st.expander("📖 **Step-by-Step Calculation Details (ICP / CPT Method)**", expanded=False):
        st.markdown(f"""
        #### 1. Pile Base Area ($A_b$)
        $$A_b = \\frac{{\\pi}}{{4}} D_0^2 = \\frac{{\\pi}}{{4}} ({D_0:.2f})^2 = {A_b_cpt:.4f} \\text{{ m}}^2$$

        #### 2. CPT End Bearing Resistance ($q_b$)
        * Cone equivalent diameter: $d_c = {d_cone} \\text{{ mm}}$
        * Base Capacity Ratio:
          $$\\frac{{q_b}}{{q_c}} = 1 - 0.5 \\log_{{10}}\\left(\\frac{{D_0 \\cdot 1000}}{{d_c}}\\right) = 1 - 0.5 \\log_{{10}}\\left(\\frac{{{D_0*1000:.0f}}}{{{d_cone}}}\\right) = {qb_qc_ratio:.3f}$$
        * Unit End Bearing: $q_b = {qb_qc_ratio:.3f} \\times {qc_tip_book:.2f} = {q_b_cpt:.2f} \\text{{ MPa}}$
        * Base Capacity ($Q_b$):
          $$Q_b = q_b \\cdot A_b \\cdot 1000 = {q_b_cpt:.2f} \\times {A_b_cpt:.4f} \\times 1000 = \\mathbf{{{Q_b_ex2:.0f} \\text{{ kN}}}}$$

        #### 3. Unit Shaft Friction ($\tau_s$) & Integration
        $$\\tau_s(z) = \\left(\\frac{{q_c}}{{45}}\\right) \\left(\\frac{{\\sigma'_v}}{{100}}\\right)^{{0.13}} \\left(\\frac{{D_0}}{{z}}\\right)^{{0.38}} \\tan(\\delta_{{cv}})$$
        * Shaft Friction Integral ($\int \\tau_s dz$): **{int_tau_mtd:.2f} kN/m**
        * Total Shaft Capacity ($Q_s$):
          $$Q_s = \\pi D_0 \\int \\tau_s dz = \\pi \\times {D_0:.2f} \\times {int_tau_mtd:.2f} = \\mathbf{{{Q_s_mtd:.0f} \\text{{ kN}}}}$$

        #### 4. Total Capacity & Number of Piles
        * Total Capacity: $Q_u = Q_b + Q_s = {Q_b_ex2:.0f} + {Q_s_mtd:.0f} = \\mathbf{{{Q_u_mtd:.0f} \\text{{ kN}}}}$$
        * Required Piles: $N = \\frac{{P_{{axial}} \\cdot 1000}}{{Q_u}} = \\frac{{{P_axial*1000:.0f}}}{{{Q_u_mtd:.0f}}} = \\mathbf{{{N_piles_mtd:.2f}}}$
        """)

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
# STEP 2: BROMS STATIC METHOD
# =========================================================
with tab_ex1:
    st.subheader("6.2.1 Preliminary Design under Static Loading (Broms 1966)")
    
    H_top = 8.0  
    A_b = (np.pi / 4) * (D_0 ** 2)
    
    sigma_b_eff = sigma_v0[-1]
    N_q = 40  
    Q_b_ex1 = A_b * sigma_b_eff * (N_q - 1)

    if "Steel" in pile_type:
        K_s1, K_s2 = 0.5, 1.0
        delta_cv = 20.0
        st.info("ℹ️ **Broms (1966) Steel Parameters:** $\\delta_{cv} = 20^\\circ$, $K_{s1}=0.5$, $K_{s2}=1.0$")
    else:
        K_s1, K_s2 = 1.0, 2.0
        delta_cv = 26.25
        st.info("ℹ️ **Broms (1966) Concrete Parameters:** $\\delta_{cv} = 26.25^\\circ$, $K_{s1}=1.0$, $K_{s2}=2.0$")

    gamma_sub1 = 17.0 - 10.0  
    gamma_sub2 = 19.0 - 10.0  
    tan_delta = np.tan(np.radians(delta_cv))

    int_sigma_v1 = 0.5 * gamma_sub1 * (H_top ** 2)
    int_sigma_v2 = 0.5 * gamma_sub2 * (L_p ** 2 - H_top ** 2)

    Q_s_layer1 = K_s1 * tan_delta * int_sigma_v1
    Q_s_layer2 = K_s2 * tan_delta * int_sigma_v2

    Q_s_ex1 = np.pi * D_0 * (Q_s_layer1 + Q_s_layer2)
    Q_u_ex1 = Q_b_ex1 + Q_s_ex1
    N_required_ex1 = (P_axial * 1000) / Q_u_ex1

    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    res_col1.metric("Base Capacity ($Q_b$)", f"{Q_b_ex1:.0f} kN")
    res_col2.metric("Shaft Capacity ($Q_s$)", f"{Q_s_ex1:.0f} kN")
    res_col3.metric("Total Capacity ($Q_u$)", f"{Q_u_ex1:.0f} kN")
    res_col4.metric("Req. Piles (FOS=1)", f"{N_required_ex1:.2f}")

    with st.expander("📖 **Step-by-Step Calculation Details (Broms 1966 Static Method)**", expanded=False):
        st.markdown(f"""
        #### 1. End Bearing Capacity ($Q_b$)
        $$Q_b = A_b \\cdot \\sigma'_b \\cdot (N_q - 1)$$
        * Area: $A_b = \\frac{{\\pi}}{{4}} ({D_0:.2f})^2 = {A_b:.4f} \\text{{ m}}^2$
        * Effective Overburden Stress at Pile Tip: $\\sigma'_b = {sigma_b_eff:.2f} \\text{{ kPa}}$
        * Bearing Capacity Factor: $N_q = {N_q}$
        * Base Capacity:
          $$Q_b = {A_b:.4f} \\times {sigma_b_eff:.2f} \\times ({N_q} - 1) = \\mathbf{{{Q_b_ex1:.0f} \\text{{ kN}}}}$$

        #### 2. Shaft Friction ($Q_s$) by Layers
        $$Q_s = \\pi D_0 \\sum \\left( K_s \\cdot \\tan\\delta_{{cv}} \\int \\sigma'_v dz \\right)$$
        * **Layer 1 (0 to {H_top}m):**
          * $\\int \\sigma'_v dz = \\frac{{1}}{{2}} \\cdot \\gamma'_{{sub1}} \\cdot H_1^2 = \\frac{{1}}{{2}} \\times {gamma_sub1:.1f} \\times {H_top}^2 = {int_sigma_v1:.1f} \\text{{ kPa}}\\cdot\\text{{m}}$
          * Unit Shaft Force: $q_{{s1}} = {K_s1} \\times \\tan({delta_cv}^\\circ) \\times {int_sigma_v1:.1f} = {Q_s_layer1:.2f} \\text{{ kN/m}}$
        * **Layer 2 ({H_top}m to {L_p}m):**
          * $\\int \\sigma'_v dz = \\frac{{1}}{{2}} \\cdot \\gamma'_{{sub2}} \\cdot (L_p^2 - H_1^2) = \\frac{{1}}{{2}} \\times {gamma_sub2:.1f} \\times ({L_p}^2 - {H_top}^2) = {int_sigma_v2:.1f} \\text{{ kPa}}\\cdot\\text{{m}}$
          * Unit Shaft Force: $q_{{s2}} = {K_s2} \\times \\tan({delta_cv}^\\circ) \\times {int_sigma_v2:.1f} = {Q_s_layer2:.2f} \\text{{ kN/m}}$
        * **Total Shaft Capacity:**
          $$Q_s = \\pi \\times {D_0:.2f} \\times ({Q_s_layer1:.2f} + {Q_s_layer2:.2f}) = \\mathbf{{{Q_s_ex1:.0f} \\text{{ kN}}}}$$

        #### 3. Total Capacity ($Q_u$)
        $$Q_u = Q_b + Q_s = {Q_b_ex1:.0f} + {Q_s_ex1:.0f} = \\mathbf{{{Q_u_ex1:.0f} \\text{{ kN}}}}$$
        """)

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

    with st.expander("📖 **Step-by-Step Calculation Details (Dynamic Soil Stiffness)**", expanded=False):
        st.markdown(f"""
        #### 1. Mean Confining Stress ($p'$) & Initial Shear Modulus ($G_0$)
        * $p' = \\frac{{1 + 2 K_0}}{{3}} \\cdot \\sigma'_{{v0, 4m}} = \\frac{{1 + 2({K_0})}}{{3}} \\times {sigma_v0_4m:.2f} = \\mathbf{{{p_prime:.2f} \\text{{ kPa}}}}$
        * $G_0 = 100 \\left(\\frac{{(3 - e)^2}}{{1 + e}}\\right) \\sqrt{{\\frac{{p'}}{{1000}}}} = 100 \\left(\\frac{{(3 - {e_silty:.2f})^2}}{{1 + {e_silty:.2f}}}\\right) \\sqrt{{\\frac{{{p_prime:.2f}}}{{1000}}}} = \\mathbf{{{G_0:.2f} \\text{{ MPa}}}}$

        #### 2. Design Shear Stress ($\tau_{{max}}$) & Shear Strain ($\gamma$)
        * Depth Reduction Factor ($r_d$ at $z=4\\text{{m}}$): $r_d = 1 - 0.01(4) = {r_d:.2f}$
        * Seismic Shear Stress: $\\tau_{{max}} = 0.65 \\cdot a_g \\cdot \\sigma_{{v0, total}} \\cdot r_d = 0.65 \\times {a_g} \\times 68.0 \\times {r_d:.2f} = \\mathbf{{{tau_max:.2f} \\text{{ kPa}}}}$
        * Solved Shear Strain ($\gamma$ via Hyperbolic Model): $\\gamma = \\mathbf{{{gamma_percent:.4f} \\%}}$

        #### 3. Secant Modulus ($G_s$, $E_s$) & Wave Velocity ($v_s$)
        * Modulus Reduction Ratio: $\\frac{{G_s}}{{G_0}} = \\frac{{1}}{{\\left(1 + \\frac{{\\gamma}}{{\\gamma_r}}\\right)^c}} = \\mathbf{{{G_ratio*100:.2f} \\%}}$
        * Secant Shear Modulus: $G_s = {G_ratio:.4f} \\times {G_0:.2f} = \\mathbf{{{G_s:.2f} \\text{{ MPa}}}}$
        * Young's Modulus ($\nu=0.5$): $E_s = 2 G_s (1 + \\nu) = 3 G_s = \\mathbf{{{E_s:.2f} \\text{{ MPa}}}}$
        * Shear Wave Velocity: $v_s = \\sqrt{{\\frac{{G_s}}{\\rho}}} = \\sqrt{{\\frac{{{G_s*1e6:.0f}}}{{1700}}}} = \\mathbf{{{v_s:.2f} \\text{{ m/s}}}}$
        * Natural Frequency: $f_n = \\frac{{v_s}}{{4 H_1}} = \\frac{{{v_s:.2f}}}{{4 \\times {H_top}}} = \\mathbf{{{f_n:.2f} \\text{{ Hz}}}}$
        """)

# =========================================================
# STEP 4: ACTIVE LENGTH & FLEXIBILITY
# =========================================================
with tab_ex4:
    st.subheader(f"6.3.2 Effective Active Length and Pile Flexibility ({pile_type})")

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

    with st.expander("📖 **Step-by-Step Calculation Details (Active Length & Flexibility)**", expanded=False):
        st.markdown(f"""
        #### 1. Pile Cross-sectional Properties
        * Inner Diameter: $D_i = D_0 - 2t = {D_0:.3f} - 2({t_wall:.3f}) = {D_i:.3f} \\text{{ m}}$
        * Moment of Inertia: $I_p = \\frac{{\\pi}}{{64}} (D_0^4 - D_i^4) = \\mathbf{{{I_p:.6f} \\text{{ m}}^4}}$
        * Flexural Rigidity: $E_p I_p = {E_pile} \\times 10^9 \\times {I_p:.6f} = \\mathbf{{{E_I/1e6:.2f} \\text{{ MN}}\\cdot\\text{{m}}^2}}$

        #### 2. Effective Active Length ($L_{{ad}}$)
        $$L_{{ad}} = 2 D_0 \\left( \\frac{{E_p}}{{E_{{sD}}}} \\right)^n$$
        * Soil Modulus at Active Depth ($10 D_0 = {10*D_0:.2f}\\text{{m}}$): $E_{{sD}} = \\mathbf{{{E_sD:.2f} \\text{{ MPa}}}}$
        * Assigned Exponent: $n = {exponent}$
        * Active Length:
          $$L_{{ad}} = 2({D_0:.2f}) \\left( \\frac{{{E_p_corrected \\times 10^9}}}{{{E_sD \\times 10^6}}} \\right)^{{{exponent}}} = \\mathbf{{{L_ad:.2f} \\text{{ m}}}}$$

        #### 3. Pile Flexibility Criteria ($T$ and $Z = L/T$)
        $$T = \\left( \\frac{{E_p I_p}}{{k_h}} \\right)^{{0.2}}$$
        * **Lower Bound ($k_h = 200 \\text{{ kN/m}}^3$):**
          * $T_u = \\left( \\frac{{{E_I:.0f}}}{{200,000}} \\right)^{{0.2}} = {T_u:.3f} \\text{{ m}}$
          * Relative Length: $Z_L = \\frac{{{L_p}}}{{{T_u:.3f}}} = \\mathbf{{{Z_L:.2f}}}$ $\\rightarrow$ **{classify_behavior(Z_L)}**
        * **Upper Bound ($k_h = 2000 \\text{{ kN/m}}^3$):**
          * $T_l = \\left( \\frac{{{E_I:.0f}}}{{2,000,000}} \\right)^{{0.2}} = {T_l:.3f} \\text{{ m}}$
          * Relative Length: $Z_u = \\frac{{{L_p}}}{{{T_l:.3f}}} = \\mathbf{{{Z_u:.2f}}}$ $\\rightarrow$ **{classify_behavior(Z_u)}**
        """)
