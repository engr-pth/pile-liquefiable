import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pile Foundation Design WorkFlow", layout="wide")

st.title("🏗️ Geotechnical & Foundation Design Workflow")
st.caption("CPT Analysis ➔ Static Capacity ➔ Dynamic Stiffness ➔ SSI ➔ Inertial Loading ➔ Kinematic Interaction ➔ Liquefaction Potential ➔ Pile Sizing ➔ liquefied ground response ➔ Lateral Spreading Analysis on Pile Group")

# ---------------------------------------------------------
# Design Procedure Flowchart Section
# ---------------------------------------------------------
# 2. Flowchart Section (နေရာမှန်: Input Parameters မတိုင်မီ တင်ပြခြင်း)
with st.expander("🗺️ **View Design Procedure Flowchart (Click to Expand)**", expanded=False):
    st.markdown("""
    ```mermaid
    flowchart TD
        A[CPT data / Structural design/forces] --> B[Evaluate soil properties<br/><i>Dr, γs, σv0, σ'v0, ϕ etc.</i>]
        B -- "Depth of full liquefaction & r_u profile" --> C[Evaluate liquefaction potential]
        C -- "Range of designs satisfying axial criteria" --> D[Determine potential foundation designs<br/>based on axial considerations:<br/>• Static considerations<br/>• Bearing failure<br/>• Settlement limit<br/>• Instability]
        
        D --> E{Sloping ground?}
        E -- Yes --> F[Evaluate extent of lateral spreading]
        E -- No --> G[Evaluate seismic loads<br/><i>including spreading</i><br/>& soil resistances]
        F --> G
        
        G -- "Seismic loads" --> H[Analyse potential designs against:<br/>• Yield/shear failure<br/>• Limiting displacement]
        H --> I[<b>Optimal design satisfying<br/>lateral & axial criteria</b>]
    ```
    """)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
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
            st.caption("ℹ️ **Steel Pipe:** $E_p = 210$ GPa, $\\delta_{cv} = 20^\\circ$")
        else:
            t_wall = st.number_input("Wall Thickness, $t$ (m)", value=0.090, step=0.005, format="%.3f")
            fc_prime = st.number_input("Concrete Strength, $f'_c$ (MPa)", value=60.0, step=5.0)
            E_pile = (4700 * np.sqrt(fc_prime)) / 1000.0  # GPa
            delta_cv = 26.25 # Concrete-Soil Interface Friction Angle
            st.caption(f"ℹ️ **Spun Concrete:** $E_p = {E_pile:.2f}$ GPa, $\\delta_{{cv}} = 26.25^\\circ$")

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
            num_rows="dynamic",
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

# Shared Variables across tabs
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

H_top = 8.0
v_s = np.sqrt((G_s_D0 * 1e6) / 1700.0)
f_n = v_s / (4 * H_top)

# ---------------------------------------------------------
# Tabs Section
# ---------------------------------------------------------
tab_ex2, tab_ex1, tab_ex3, tab_ex4, tab_ex5, tab_ex6, tab_ex7, tab_ex8, tab_ex9, tab_ex10 = st.tabs([
    "📊 Step 1: CPT Capacity (Ex 2)", 
    "📌 Step 2: Broms Static Capacity (Ex 1)", 
    "🌊 Step 3: Soil Stiffness (Ex 3)",
    "📏 Step 4: SSI & Active Length (Ex 4)",
    "💥 Step 5: Inertial Loading (Ex 5)",
    "🔄 Step 6: Kinematic Interaction (Ex 6)",
    "🌋 Step 7: Liquefaction Potential (Ex 7)",
    "🎯 Step 8: Pile Sizing based on Liquefaction considerations (Ex 8)",
    "🌊 Step 9: Liquefied Ground Response (Ex 9)",
    "📏 Step 10: Lateral Spreading Analysis on Pile Group (Ex 10)"
])

# =========================================================
# STEP 1: CPT METHOD
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
        st.write(r"#### 1. Pile Base Area ($A_b$)")
        st.latex(rf"A_b = \frac{{\pi}}{{4}} D_0^2 = \frac{{\pi}}{{4}} ({D_0:.2f})^2 = {A_b_cpt:.4f} \text{{ m}}^2")

        st.write(r"#### 2. CPT End Bearing Resistance ($q_b$)")
        st.write(f"* Cone equivalent diameter: $d_c = {d_cone} \\text{{ mm}}$")
        st.latex(rf"\frac{{q_b}}{{q_c}} = 1 - 0.5 \log_{{10}}\left(\frac{{D_0 \cdot 1000}}{{d_c}}\right) = 1 - 0.5 \log_{{10}}\left(\frac{{{D_0*1000:.0f}}}{{{d_cone}}}\right) = {qb_qc_ratio:.3f}")
        st.latex(rf"q_b = {qb_qc_ratio:.3f} \times {qc_tip_book:.2f} = {q_b_cpt:.2f} \text{{ MPa}}")
        st.latex(rf"Q_b = q_b \cdot A_b \cdot 1000 = {q_b_cpt:.2f} \times {A_b_cpt:.4f} \times 1000 = {Q_b_ex2:.0f} \text{{ kN}}")

        st.write(r"#### 3. Unit Shaft Friction ($\tau_s$) & Integration")
        st.latex(r"\tau_s(z) = \left(\frac{q_c}{45}\right) \left(\frac{\sigma'_v}{100}\right)^{0.13} \left(\frac{D_0}{z}\right)^{0.38} \tan(\delta_{cv})")
        st.write(f"* Shaft Friction Integral ($\\int \\tau_s dz$): **{int_tau_mtd:.2f} kN/m**")
        st.latex(rf"Q_s = \pi D_0 \int \tau_s dz = \pi \times {D_0:.2f} \times {int_tau_mtd:.2f} = {Q_s_mtd:.0f} \text{{ kN}}")

        st.write(r"#### 4. Total Capacity & Number of Piles")
        st.latex(rf"Q_u = Q_b + Q_s = {Q_b_ex2:.0f} + {Q_s_mtd:.0f} = {Q_u_mtd:.0f} \text{{ kN}}")
        st.latex(rf"N = \frac{{P_{{axial}} \cdot 1000}}{{Q_u}} = \frac{{{P_axial*1000:.0f}}}{{{Q_u_mtd:.0f}}} = {N_piles_mtd:.2f}")

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
        st.write(r"#### 1. End Bearing Capacity ($Q_b$)")
        st.latex(r"Q_b = A_b \cdot \sigma'_b \cdot (N_q - 1)")
        st.write(f"* Area: $A_b = \\frac{{\\pi}}{{4}} ({D_0:.2f})^2 = {A_b:.4f} \\text{{ m}}^2$")
        st.write(f"* Effective Overburden Stress at Pile Tip: $\\sigma'_b = {sigma_b_eff:.2f} \\text{{ kPa}}$")
        st.write(f"* Bearing Capacity Factor: $N_q = {N_q}$")
        st.latex(rf"Q_b = {A_b:.4f} \times {sigma_b_eff:.2f} \times ({N_q} - 1) = {Q_b_ex1:.0f} \text{{ kN}}")

        st.write(r"#### 2. Shaft Friction ($Q_s$) by Layers")
        st.latex(r"Q_s = \pi D_0 \sum \left( K_s \cdot \tan\delta_{cv} \int \sigma'_v dz \right)")
        st.write(f"**Layer 1 (0 to {H_top}m):**")
        st.latex(rf"\int \sigma'_v dz = \frac{{1}}{{2}} \cdot \gamma'_{{sub1}} \cdot H_1^2 = \frac{{1}}{{2}} \times {gamma_sub1:.1f} \times {H_top}^2 = {int_sigma_v1:.1f} \text{{ kPa}}\cdot\text{{m}}")
        st.latex(rf"q_{{s1}} = {K_s1} \times \tan({delta_cv}^\circ) \times {int_sigma_v1:.1f} = {Q_s_layer1:.2f} \text{{ kN/m}}")
        st.write(f"**Layer 2 ({H_top}m to {L_p}m):**")
        st.latex(rf"\int \sigma'_v dz = \frac{{1}}{{2}} \cdot \gamma'_{{sub2}} \cdot (L_p^2 - H_1^2) = \frac{{1}}{{2}} \times {gamma_sub2:.1f} \times ({L_p}^2 - {H_top}^2) = {int_sigma_v2:.1f} \text{{ kPa}}\cdot\text{{m}}")
        st.latex(rf"q_{{s2}} = {K_s2} \times \tan({delta_cv}^\circ) \times {int_sigma_v2:.1f} = {Q_s_layer2:.2f} \text{{ kN/m}}")
        st.write("**Total Shaft Capacity:**")
        st.latex(rf"Q_s = \pi \times {D_0:.2f} \times ({Q_s_layer1:.2f} + {Q_s_layer2:.2f}) = {Q_s_ex1:.0f} \text{{ kN}}")

        st.write(r"#### 3. Total Capacity ($Q_u$)")
        st.latex(rf"Q_u = Q_b + Q_s = {Q_b_ex1:.0f} + {Q_s_ex1:.0f} = {Q_u_ex1:.0f} \text{{ kN}}")

# =========================================================
# STEP 3: DYNAMIC SOIL STIFFNESS
# =========================================================
with tab_ex3:
    st.subheader("6.3.1 Soil Stiffness and Natural Frequency")
    
    gamma_percent = gamma_sol * 100
    G_s = G_ratio * G_0  # MPa
    nu = 0.5
    E_s = 2 * G_s * (1 + nu)  # MPa

    G_s_pa = G_s * 1e6  # Convert MPa to Pa
    rho_soil = 1700.0   # kg/m³
    v_s = np.sqrt(G_s_pa / rho_soil)  # m/s
    f_n = v_s / (4 * H_top)  # Hz

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
        st.write(r"#### 1. Mean Confining Stress ($p'$) & Initial Shear Modulus ($G_0$)")
        st.latex(rf"p' = \frac{{1 + 2 K_0}}{{3}} \cdot \sigma'_{{v0, 4m}} = \frac{{1 + 2({K_0})}}{{3}} \times {sigma_v0_4m:.2f} = {p_prime:.2f} \text{{ kPa}}")
        st.latex(rf"G_0 = 100 \left(\frac{{(3 - e)^2}}{{1 + e}}\right) \sqrt{{\frac{{p'}}{{1000}}}} = 100 \left(\frac{{(3 - {e_silty:.2f})^2}}{{1 + {e_silty:.2f}}}\right) \sqrt{{\frac{{{p_prime:.2f}}}{{1000}}}} = {G_0:.2f} \text{{ MPa}}")

        st.write(r"#### 2. Design Shear Stress ($\tau_{max}$) & Shear Strain ($\gamma$)")
        st.write(f"* Depth Reduction Factor ($r_d$ at $z=4\\text{{m}}$): $r_d = 1 - 0.01(4) = {r_d:.2f}$")
        st.latex(rf"\tau_{{max}} = 0.65 \cdot a_g \cdot \sigma_{{v0, total}} \cdot r_d = 0.65 \times {a_g} \times 68.0 \times {r_d:.2f} = {tau_max:.2f} \text{{ kPa}}")
        st.write(f"* Solved Shear Strain ($\\gamma$ via Hyperbolic Model): **\\gamma = {gamma_percent:.4f} %**")

        st.write(r"#### 3. Secant Modulus ($G_s$, $E_s$), Shear Wave Velocity ($v_s$) & Natural Frequency ($f_n$)")
        st.latex(rf"\frac{{G_s}}{{G_0}} = \frac{{1}}{{\left(1 + \frac{{\gamma}}{{\gamma_r}}\right)^c}} = {G_ratio*100:.2f} \%")
        st.latex(rf"G_s = {G_ratio:.4f} \times {G_0:.2f} = {G_s:.2f} \text{{ MPa}} \quad ({G_s_pa:.0f} \text{{ Pa}})")
        st.latex(rf"E_s = 2 G_s (1 + \nu) = 3 G_s = {E_s:.2f} \text{{ MPa}}")
        st.latex(rf"v_s = \sqrt{{\frac{{G_s}}{{\rho}}}} = \sqrt{{\frac{{{G_s_pa:.0f}}}{{{rho_soil:.0f}}}}} = {v_s:.2f} \text{{ m/s}}")
        st.latex(rf"f_n = \frac{{v_s}}{{4 H_1}} = \frac{{{v_s:.2f}}}{{4 \times {H_top}}} = {f_n:.2f} \text{{ Hz}}")

# =========================================================
# STEP 4: ACTIVE LENGTH & SSI
# =========================================================
with tab_ex4:
    st.subheader(f"6.3.2 Effective Active Length and Pile Flexibility ({pile_type})")

    st.markdown("---")
    st.markdown("##### ⚙️ Subgrade Modulus Gradient ($k$) Selection Mode")
    
    k_mode = st.radio(
        "Choose Analysis Approach for Subgrade Modulus Gradient (k):",
        [
            "Option 1: Single Nominal Value (Auto CPT qc / Manual)",
            "Option 2: Bounding Approach (Upper & Lower Bound Range)"
        ],
        horizontal=True
    )

    L_ad = 2 * D_0 * ((E_p_corrected * 1e9) / (E_sD * 1e6)) ** exponent
    E_I = (E_pile * 1e9) * I_p

    def classify_behavior(z_val):
        if z_val > 5.0: return "Flexible"
        elif 2.5 <= z_val <= 5.0: return "Semi-Flexible"
        else: return "Rigid"

    if "Option 1" in k_mode:
        k_calc_method = st.radio(
            "Determine $k$ via:",
            ["Auto-calculate from CPT Profile (0 to 5D0)", "Manual Input"],
            horizontal=True
        )

        if "Auto-calculate" in k_calc_method:
            z_active_limit = 5.0 * D_0
            active_mask_5d = depths <= z_active_limit
            qc_active_subset = qc_values[active_mask_5d]
            sigma_v_active_subset = np.array(sigma_v0)[active_mask_5d]
            
            qc_avg_active = np.mean(qc_active_subset) if len(qc_active_subset) > 0 else 1.0
            sigma_v_eff = np.mean(sigma_v_active_subset) if len(sigma_v_active_subset) > 0 else 10.0
            
            dr_est = min(100.0, max(10.0, 100 * np.sqrt((qc_avg_active * 1000) / (300 * np.sqrt(max(sigma_v_eff, 1.0))))))
            
            if dr_est < 40:
                k_nominal = 1300 + (dr_est / 40.0) * (2200 - 1300)
            elif dr_est <= 80:
                k_nominal = 2200 + ((dr_est - 40) / 40.0) * (5400 - 2200)
            else:
                k_nominal = 5400 + ((dr_est - 80) / 20.0) * (11000 - 5400)
                
            st.info(
                f"📊 **Auto-fetched Active Depth ($0 - {z_active_limit:.2f}\\text{{ m}}$):** "
                f"Avg $q_c = \\mathbf{{{qc_avg_active:.2f}\\text{{ MPa}}}}$ | "
                f"Est. $D_r = \\mathbf{{{dr_est:.1f}\\%}}$ $\\rightarrow$ Derived $k = \\mathbf{{{k_nominal:.0f}\\text{{ kN/m³}}}}$ (API RP 2GEO)"
            )
        else:
            k_nominal = st.number_input("Enter Gradient of Soil Modulus, $k$ (kN/m³)", value=2200.0, step=100.0)

        T_nom = ((E_I) / (k_nominal * 1e3)) ** 0.2
        Z_nom = L_p / T_nom

        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        col_e1.metric("Design Pile Modulus ($E_p$)", f"{E_p_corrected:.1f} GPa")
        col_e2.metric(f"Soil Modulus ($E_{{sD}}$)", f"{E_sD:.2f} MPa")
        col_e3.metric("Effective Active Length ($L_{ad}$)", f"{L_ad:.2f} m")
        col_e4.metric("Flexural Rigidity ($E_p I_p$)", f"{E_I/1e6:.1f} MN·m²")

        col_e5, col_e6 = st.columns(2)
        col_e5.metric("Elastic Length ($T$)", f"{T_nom:.3f} m")
        col_e6.metric("Relative Length ($Z = L/T$)", f"{Z_nom:.2f} ({classify_behavior(Z_nom)})")

    else:
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            k_lower = st.number_input("Lower Bound $k_{lower}$ (kN/m³)", value=200.0, step=50.0)
        with col_k2:
            k_upper = st.number_input("Upper Bound $k_{upper}$ (kN/m³)", value=2000.0, step=100.0)

        T_u = ((E_I) / (k_lower * 1e3)) ** 0.2
        Z_L = L_p / T_u
        T_l = ((E_I) / (k_upper * 1e3)) ** 0.2
        Z_u = L_p / T_l

        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        col_e1.metric("Design Pile Modulus ($E_p$)", f"{E_p_corrected:.1f} GPa")
        col_e2.metric(f"Soil Modulus ($E_{{sD}}$)", f"{E_sD:.2f} MPa")
        col_e3.metric("Effective Active Length ($L_{ad}$)", f"{L_ad:.2f} m")
        col_e4.metric("Flexural Rigidity ($E_p I_p$)", f"{E_I/1e6:.1f} MN·m²")

        col_e5, col_e6, col_e7, col_e8 = st.columns(4)
        col_e5.metric("Elastic Length ($T_u$, Lower)", f"{T_u:.3f} m")
        col_e6.metric("Relative Length ($Z_L$)", f"{Z_L:.2f} ({classify_behavior(Z_L)})")
        col_e7.metric("Elastic Length ($T_l$, Upper)", f"{T_l:.3f} m")
        col_e8.metric("Relative Length ($Z_u$)", f"{Z_u:.2f} ({classify_behavior(Z_u)})")

    with st.expander("📖 Step-by-Step Calculation Details (Tab 4: Active Length & Flexibility)"):
        st.markdown("### 1. Pile Flexural Rigidity ($E_p I_p$) & Diameter Correction")
        st.latex(r"I_p = \frac{\pi}{64} \left( D_0^4 - D_i^4 \right)")
        if "Steel" in pile_type:
            st.write(f"* **Inner Diameter ($D_i$):** $D_0 - 2t = {D_0:.3f} - 2({t_wall:.3f}) = {D_i:.3f}$ m")
            st.write(f"* **Moment of Inertia ($I_p$):** ${I_p:.6f}$ m⁴")
            st.latex(r"E_{p,\text{corrected}} = \frac{E_p}{\frac{D_0^4}{D_0^4 - D_i^4}}")
            st.write(f"* **Corrected Pile Elastic Modulus ($E_{{p,corrected}}$):** ${E_p_corrected:.2f}$ GPa")
        else:
            st.write(f"* **Solid Pile Section ($I_p$):** ${I_p:.6f}$ m⁴")
            st.write(f"* **Design Pile Modulus ($E_p$):** ${E_p_corrected:.2f}$ GPa")

        st.markdown("---")
        st.markdown("### 2. Operational Soil Shear Modulus ($G_{sD}$) and Young's Modulus ($E_{sD}$)")
        st.latex(r"\sigma'_{v0} = 7.0 \times D_0")
        st.latex(r"p' = \left(\frac{1 + 2K_0}{3}\right) \sigma'_{v0}")
        st.latex(r"G_0 = 100 \cdot \frac{(3 - e)^2}{1 + e} \cdot \sqrt{\frac{p'}{1000}} \quad (\text{MPa})")
        st.latex(r"G_{sD} = \left(\frac{G}{G_0}\right) \times G_0, \quad E_{sD} = 3 \times G_{sD}")
        
        st.write(f"* **Effective Stress at $D_0$ ($\sigma'_{{v0}}$):** ${sigma_v0_D0:.2f}$ kPa")
        st.write(f"* **Mean Effective Stress ($p'$):** ${p_prime_D0:.2f}$ kPa")
        st.write(f"* **Small-Strain Shear Modulus ($G_0$):** ${G_0_D0:.2f}$ MPa")
        st.write(f"* **Degraded Shear Modulus ($G_{{sD}}$):** ${G_s_D0:.2f}$ MPa")
        st.write(f"* **Operational Soil Modulus ($E_{{sD}}$):** ${E_sD:.2f}$ MPa")

        st.markdown("---")
        st.markdown("### 3. Effective Active Length ($L_{ad}$)")
        st.latex(r"L_{ad} = 2 \cdot D_0 \cdot \left( \frac{E_p}{E_{sD}} \right)^n")
        
        ratio_val = (E_p_corrected * 1000.0) / E_sD
        st.write(f"* **Selected Exponent ($n$):** ${exponent:.2f}$")
        st.write(f"* **Effective Active Depth ($L_{{ad}}$):** $2 \\times {D_0:.2f} \\times ({ratio_val:.1f})^{{{exponent:.2f}}} = \\mathbf{{{L_ad:.2f}\\text{{ m}}}}$")

        st.markdown("---")
        st.markdown("### 4. Subgrade Modulus Gradient ($k$) & Relative Length ($Z = L/T$)")
        
        if "Option 1" in k_mode:
            st.latex(r"T = \left( \frac{E_p I_p}{k} \right)^{0.2}")
            st.latex(r"Z = \frac{L_p}{T}")
            st.write(f"* **Design Modulus Gradient ($k$):** ${k_nominal:.0f}$ kN/m³")
            st.write(f"* **Characteristic Elastic Length ($T$):** ${T_nom:.3f}$ m")
            st.write(f"* **Relative Length Ratio ($Z$):** ${Z_nom:.2f}$ $\\rightarrow$ **{classify_behavior(Z_nom)} Behavior**")
        else:
            st.latex(r"T_u = \left( \frac{E_p I_p}{k_{\text{lower}}} \right)^{0.2}, \quad T_l = \left( \frac{E_p I_p}{k_{\text{upper}}} \right)^{0.2}")
            st.write(f"* **Lower Bound $k_{{lower}}$:** ${k_lower:.0f}$ kN/m³ $\\rightarrow T_u = {T_u:.3f}$ m, $Z_L = {Z_L:.2f}$ ({classify_behavior(Z_L)})")
            st.write(f"* **Upper Bound $k_{{upper}}$:** ${k_upper:.0f}$ kN/m³ $\\rightarrow T_l = {T_l:.3f}$ m, $Z_u = {Z_u:.2f}$ ({classify_behavior(Z_u)})")

        st.markdown("---")
        st.markdown("### 5. Classification Criteria (Flexible vs. Rigid)")
        st.markdown("""
        * **$Z > 5.0$:** Flexible Pile Behavior (Pile Top Deflection is Independent of Tip Boundary Conditions).
        * **$2.5 \le Z \le 5.0$:** Semi-Flexible / Intermediate Behavior.
        * **$Z < 2.5$:** Rigid Pile Behavior (Short Stubby Pile Rotation).
        """)

# =========================================================
# STEP 5: INERTIAL LOADING ON THE PILE
# =========================================================
with tab_ex5:
    st.subheader("6.3.3 Inertial Loading on the Pile (Eurocode 8)")

    st.markdown("##### ⚙️ Superstructure & Dynamic Input Parameters")
    col_in1, col_in2, col_in3, col_in4 = st.columns(4)
    with col_in1:
        n_piles = st.number_input("Number of Piles in Group, $N_{group}$", value=4, step=1)
    with col_in2:
        m_super = st.number_input("Superstructure Mass, $m$ (tons)", value=940.0, step=10.0)
    with col_in3:
        h_mass = st.number_input("Height of Mass Center, $h$ (m)", value=10.0, step=0.5)
    with col_in4:
        spectral_acc_factor = st.number_input("Spectral Acc. Ratio ($S_{da} / a_g$)", value=2.13, step=0.01)

    stiffness_ratio = (E_p_corrected * 1000.0) / E_sD

    K_HH_single = 0.79 * D_0 * (E_sD * 1e6) * (stiffness_ratio ** 0.28)
    K_MM_single = 0.15 * (D_0 ** 3) * (E_sD * 1e6) * (stiffness_ratio ** 0.77)
    K_HM_single = -0.24 * (D_0 ** 2) * (E_sD * 1e6) * (stiffness_ratio ** 0.53)

    e_eccentricity = K_HM_single / K_HH_single
    K_h_eq_single = (K_HH_single * K_MM_single - K_HM_single ** 2) / (K_MM_single - e_eccentricity * K_HM_single)

    K_h_group = n_piles * K_h_eq_single
    mass_kg = m_super * 1000.0  # kg
    f_p_group = (1.0 / (2 * np.pi)) * np.sqrt(K_h_group / mass_kg)
    T_p_group = 1.0 / f_p_group if f_p_group > 0 else 0.0

    a_response = spectral_acc_factor * a_g * 9.81  # m/s²
    H_inertial_MN = (mass_kg * a_response) / 1e6    # MN
    M_inertial_MNm = H_inertial_MN * h_mass         # MN·m
    delta_h_m = H_inertial_MN / (K_h_group / 1e6)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stiffness Ratio ($E_p / E_{sD}$)", f"{stiffness_ratio:.1f}")
    m2.metric("Equivalent $K_h$ (per pile)", f"{K_h_eq_single/1e6:.1f} MN/m")
    m3.metric("Natural Freq. ($f_p$)", f"{f_p_group:.2f} Hz")
    m4.metric("Natural Period ($T_p$)", f"{T_p_group:.2f} s")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Response Acc. ($a_{resp}$)", f"{a_response:.2f} m/s²")
    m6.metric("Inertial Shear ($H$)", f"{H_inertial_MN:.2f} MN")
    m7.metric("Inertial Moment ($M$)", f"{M_inertial_MNm:.1f} MN·m")
    m8.metric("Peak Deflection ($\delta_h$)", f"{delta_h_m*1000:.1f} mm")

    with st.expander("📖 **Step-by-Step Calculation Details (Inertial Loading & Stiffness Matrix)**", expanded=False):
        st.write(r"#### 1. Pile-to-Soil Stiffness Ratio")
        st.latex(rf"\frac{{E_p}}{{E_{{sD}}}} = \frac{{{E_p_corrected:.2f} \times 10^9}}{{{E_sD:.2f} \times 10^6}} = {stiffness_ratio:.1f}")

        st.write(r"#### 2. Head Stiffness Coefficients (Eurocode 8 - Square Root Variation)")
        st.latex(rf"K_{{HH}} = 0.79 \cdot D_0 \cdot E_{{sD}} \left(\frac{{E_p}}{{E_{{sD}}}}\right)^{{0.28}} = {K_HH_single/1e6:.1f} \text{{ MN/m}}")
        st.latex(rf"K_{{MM}} = 0.15 \cdot D_0^3 \cdot E_{{sD}} \left(\frac{{E_p}}{{E_{{sD}}}}\right)^{{0.77}} = {K_MM_single/1e6:.1f} \text{{ MNm/rad}}")
        st.latex(rf"K_{{HM}} = -0.24 \cdot D_0^2 \cdot E_{{sD}} \left(\frac{{E_p}}{{E_{{sD}}}}\right)^{{0.53}} = {K_HM_single/1e6:.1f} \text{{ MN}}")

        st.write(r"#### 3. Eccentricity ($e$) for Rigid Pile Cap (Fixed-Head Condition)")
        st.latex(rf"e = \frac{{K_{{HM}}}}{{K_{{HH}}}} = \frac{{{K_HM_single / 1e6:.1f}}}{{{K_HH_single / 1e6:.1f}}} = {e_eccentricity:.2f} \text{{ m}}")

        st.write(r"#### 4. Equivalent Horizontal Head Stiffness ($K_h$)")
        st.latex(rf"K_h = \frac{{K_{{HH}} K_{{MM}} - K_{{HM}}^2}}{{K_{{MM}} - e K_{{HM}}}} = {K_h_eq_single/1e6:.1f} \text{{ MN/m}}")

        st.write(r"#### 5. Pile Group Natural Frequency ($f_p$) and Response Acceleration")
        st.latex(rf"f_p = \frac{{1}}{{2\pi}} \sqrt{{\frac{{N_{{group}} \cdot K_h}}{{m}}}} = \frac{{1}}{{2\pi}} \sqrt{{\frac{{{n_piles} \times {K_h_eq_single/1e6:.1f} \times 10^6}}{{{mass_kg:.0f}}}}} = {f_p_group:.2f} \text{{ Hz}} \quad (T_p = {T_p_group:.2f} \text{{ s}})")
        st.latex(rf"a_{{response}} = \left(\frac{{S_{{da}}}}{{a_g}}\right) \cdot a_g = {spectral_acc_factor} \times {a_g:.2f}\text{{g}} = {a_response:.2f} \text{{ m/s}}^2")

        st.write(r"#### 6. Inertial Load & Displacement Calculations")
        st.latex(rf"H = m \cdot a_{{response}} = {m_super:.0f} \times 10^3 \times {a_response:.2f} = {H_inertial_MN:.2f} \text{{ MN}}")
        st.latex(rf"M = H \cdot h = {H_inertial_MN:.2f} \times {h_mass:.1f} = {M_inertial_MNm:.1f} \text{{ MN}}\cdot\text{{m}}")
        st.latex(rf"\delta_h = \frac{{H}}{{N_{{group}} \cdot K_h}} = \frac{{{H_inertial_MN:.2f}}}{{{n_piles} \times {K_h_eq_single/1e6:.1f}}} = {delta_h_m:.3f} \text{{ m}} \quad ({delta_h_m*1000:.1f} \text{{ mm}})")

# =========================================================
# STEP 6: KINEMATIC INTERACTION (GAZETAS, 1984)
# =========================================================
with tab_ex6:
    st.subheader("6.3.4 Kinematic Interaction Analysis (Gazetas, 1984)")

    col_k1, col_k2 = st.columns([1.2, 1.0])

    with col_k1:
        st.markdown("##### ⚙️ Kinematic Parameters")
        profile_type = st.selectbox(
            "Soil Stiffness Variation Profile",
            ["Parabolic Variation", "Constant Stiffness", "Linear Increase"],
            index=0
        )

        u_0_input = st.number_input("Peak Free-Field Surface Displacement, $u_0$ (mm)", value=92.0, step=1.0)
        
        st.markdown("###### 🌐 Static Ground Displacement ($u_g$ / Bedrock PGD at $f=0$ Hz)")
        ug_method = st.selectbox(
            "Method to determine $u_g$",
            [
                "1. Direct Input (1D Site Response / DEEPSOIL / SHAKE)",
                "2. Eurocode 8 PGD Formula (d_g = 0.025 * a_g * S * T_C * T_D)",
                "3. NGA Attenuation / Ratio Estimation (u_g = Ratio * u_0)"
            ],
            index=0
        )

        if "1. Direct Input" in ug_method:
            u_g_input = st.number_input(
                "Static Ground Displacement, $u_g$ (mm)", 
                value=50.0, step=1.0,
                help="Bedrock PGD obtained from double integration of acceleration time-history motion."
            )
        elif "2. Eurocode 8" in ug_method:
            col_ec1, col_ec2 = st.columns(2)
            with col_ec1:
                pga_g = st.number_input("PGA, $a_g$ (g)", value=0.35, step=0.05)
                S_factor = st.number_input("Soil Factor, $S$", value=1.2, step=0.1)
            with col_ec2:
                T_C = st.number_input("Period $T_C$ (s)", value=0.5, step=0.05)
                T_D = st.number_input("Period $T_D$ (s)", value=2.0, step=0.1)
            
            u_g_calc = 0.025 * (pga_g * 9.81) * S_factor * T_C * T_D * 1000.0
            u_g_input = st.number_input("Calculated $u_g$ (mm)", value=float(np.round(u_g_calc, 1)), disabled=True)
        else:
            ratio_ug = st.slider("Baseline Ratio ($u_g / u_0$)", min_value=0.3, max_value=0.8, value=0.55, step=0.05)
            u_g_input = u_0_input * ratio_ug
            st.info(f"Estimated Static Baseline $u_g = {u_g_input:.1f}$ mm")

        f_p_input = st.number_input("Pile Group Natural Frequency, $f_p$ (Hz)", value=f_p_group, step=0.1)
        f_n_input = st.number_input("Soil Layer Natural Frequency, $f_n$ (Hz)", value=f_n, step=0.1)

        if profile_type == "Constant Stiffness":
            exp_Ep = 0.30
            exp_LD = -0.50
            coeff_a, coeff_b, coeff_c = 0.0, 0.0, -0.21
        elif profile_type == "Parabolic Variation":
            exp_Ep = 0.16
            exp_LD = -0.35
            coeff_a, coeff_b, coeff_c = 3.64e-6, -4.36e-4, 6.0e-3
        else:  # Linear Increase
            exp_Ep = 0.10
            exp_LD = -0.40
            coeff_a, coeff_b, coeff_c = -6.75e-5, -7.0e-3, 3.3e-2

        freq_ratio = f_p_input / f_n_input if f_n_input > 0 else 0.0
        L_D_ratio = L_p / D_0

        F_factor = freq_ratio * (stiffness_ratio ** exp_Ep) * (L_D_ratio ** exp_LD)
        I_u_calc = coeff_a * (F_factor**4) + coeff_b * (F_factor**3) + coeff_c * (F_factor**2) + 1.0
        I_u = max(I_u_calc, 0.5)

        u_p = I_u * u_0_input

    with col_k2:
        st.markdown("##### 📊 Kinematic Interaction Results")
        k_m1, k_m2 = st.columns(2)
        k_m1.metric("Dimensionless Factor ($F$)", f"{F_factor:.3f}")
        k_m2.metric("Interaction Factor ($I_u$)", f"{I_u:.3f}")

        k_m3, k_m4 = st.columns(2)
        k_m3.metric("Free-Field Displacement ($u_0$)", f"{u_0_input:.1f} mm")
        k_m4.metric("Pile Head Displacement ($u_p$)", f"{u_p:.1f} mm")

        st.metric("Static Ground Displacement ($u_g$)", f"{u_g_input:.1f} mm")

    st.markdown("---")
    st.subheader("📈 Parametric Response Curves")

    tab_plot1, tab_plot2 = st.tabs(["I_u vs Pile Frequency Curve", "Frequency Response Curve (u₀ vs uₚ)"])

    with tab_plot1:
        f_p_range = np.linspace(0.1, 10.0, 200)
        F_range = (f_p_range / f_n_input) * (stiffness_ratio ** exp_Ep) * (L_D_ratio ** exp_LD)
        I_u_range = coeff_a * (F_range**4) + coeff_b * (F_range**3) + coeff_c * (F_range**2) + 1.0
        I_u_range = np.maximum(I_u_range, 0.5)

        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(f_p_range, I_u_range, color='blue', linewidth=2, label="$I_u$ Curve")
        ax1.scatter([f_p_input], [I_u], color='red', s=70, zorder=5, label=f"Current ($f_p={f_p_input:.2f}$ Hz, $I_u={I_u:.2f}$)")
        ax1.set_xlabel("Pile Group Natural Frequency, $f_p$ (Hz)", fontsize=11)
        ax1.set_ylabel("Interaction Factor, $I_u$", fontsize=11)
        ax1.set_title(f"Variation of Interaction Factor $I_u$ with $f_p$ ({profile_type})", fontsize=12)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()
        st.pyplot(fig1)

    with tab_plot2:
        freq_axis = np.linspace(0.01, 10.0, 300)
        r = freq_axis / f_n_input
        
        alpha = 0.12
        u_base_fn = u_g_input * np.exp(-alpha * f_n_input)
        u_base = u_g_input * np.exp(-alpha * freq_axis)

        beta = 0.22
        amp_shape = (r**2) / np.sqrt((1 - r**2)**2 + (2 * beta * r)**2) * np.exp(-0.35 * r)
        
        amp_at_fn = (1.0 / (2 * beta)) * np.exp(-0.35)
        S_f = amp_shape / amp_at_fn

        u0_curve = u_base + (u_0_input - u_base_fn) * S_f

        F_curve = r * (stiffness_ratio ** exp_Ep) * (L_D_ratio ** exp_LD)
        I_u_curve = coeff_a * (F_curve**4) + coeff_b * (F_curve**3) + coeff_c * (F_curve**2) + 1.0
        I_u_curve = np.maximum(I_u_curve, 0.5)
        
        up_curve = I_u_curve * u0_curve

        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        ax2.plot(freq_axis, u0_curve, label="$u_0$ (Free-field response)", color="black", linestyle="-", linewidth=1.2)
        ax2.plot(freq_axis, up_curve, label="$u_p$ (Pile head response)", color="navy", linewidth=2.5)
        
        ax2.set_xlabel("Natural frequency (Hz)", fontsize=11, fontweight='bold')
        ax2.set_ylabel("Displacement (mm)", fontsize=11, fontweight='bold')
        ax2.set_title("Free Field and Pile Head Response due to Kinematic Interaction", fontsize=12)
        
        max_y = max(np.max(u0_curve), np.max(up_curve)) * 1.15
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, max_y)
        
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="upper right")
        st.pyplot(fig2)

    with st.expander("📖 **Step-by-Step Calculation Details (Kinematic Interaction)**", expanded=False):
        st.write(r"#### 1. Dimensionless Factor ($F$)")
        st.latex(rf"F = \left(\frac{{f_p}}{{f_n}}\right) \left(\frac{{E_p}}{{E_{{sD}}}}\right)^{{{exp_Ep}}} \left(\frac{{L}}{{D}}\right)^{{{exp_LD}}}")
        st.latex(rf"F = \left(\frac{{{f_p_input:.2f}}}{{{f_n_input:.2f}}}\right) \left({stiffness_ratio:.1f}\right)^{{{exp_Ep}}} \left(\frac{{{L_p}}}{{{D_0}}}\right)^{{{exp_LD}}} = {F_factor:.3f}")

        st.write(r"#### 2. Kinematic Interaction Factor ($I_u$)")
        st.latex(rf"I_u = a F^4 + b F^3 + c F^2 + 1.0")
        st.write(f"* Coefficients for **{profile_type}**: $a = {coeff_a}$, $b = {coeff_b}$, $c = {coeff_c}$")
        st.latex(rf"I_u = ({coeff_a})({F_factor:.3f})^4 + ({coeff_b})({F_factor:.3f})^3 + ({coeff_c})({F_factor:.3f})^2 + 1.0 = {I_u:.3f}")

        st.write(r"#### 3. Pile Head Displacement ($u_p$)")
        st.latex(rf"u_p = I_u \times u_0 = {I_u:.3f} \times {u_0_input:.1f} = {u_p:.1f} \text{{ mm}}")

# =========================================================
# STEP 7: LIQUEFACTION POTENTIAL FROM CPT DATA (EX 7)
# =========================================================
with tab_ex7:
    st.subheader("6.4.1 Example 7: Determination of Liquefaction Potential from CPT Data")
    st.caption("Seed & Idriss (1971) / Idriss & Boulanger (2004) Methodology")

    # Compute Total and Effective Overburden Stress for CPT depths
    p_a = 100.0  # Atmospheric pressure in kPa

    sigma_v0_total_list = [0.0]
    sigma_v0_eff_list = [0.0]

    for i in range(1, len(depths)):
        dz = depths[i] - depths[i-1]
        g_tot = gamma_values[i]
        g_eff = max(g_tot - 9.81, 1.0)
        sigma_v0_total_list.append(sigma_v0_total_list[-1] + g_tot * dz)
        sigma_v0_eff_list.append(sigma_v0_eff_list[-1] + g_eff * dz)

    # Idriss & Boulanger MSF (Eq 6.67)
    MSF_liq = 6.9 * np.exp(-M_w / 4.0) - 0.058

    qcn_list = []
    crr_list = []
    csr_list = []
    rd_list = []
    fs_liq_list = []
    status_list = []

    for z, qc_mpA, s_v0_tot, s_v0_eff in zip(depths, qc_values, sigma_v0_total_list, sigma_v0_eff_list):
        qc_kpa = qc_mpA * 1000.0
        qcn = qc_kpa / p_a
        qcn_list.append(qcn)

        # CRR calculation for soils with <5% fines content (Eq 6.62)
        if qcn > 0:
            crr_val = np.exp((qcn / 540.0) + (qcn / 67.0)**2 - (qcn / 80.0)**3 + (qcn / 114.0)**4 - 3.0)
        else:
            crr_val = 0.05
        crr_list.append(crr_val)

        # rd calculation (Eqs 6.64 - 6.66)
        if z == 0:
            rd_val = 1.0
        else:
            alpha_z = -1.012 - 1.126 * np.sin((z / 11.73) + 5.133)
            beta_z = 0.106 + 0.118 * np.sin((z / 11.28) + 5.142)
            rd_val = np.exp(alpha_z + beta_z * M_w)
        rd_list.append(rd_val)

        # CSR calculation (Eq 6.63)
        if z > 0 and s_v0_eff > 0:
            csr_val = 0.65 * (a_g * s_v0_tot / s_v0_eff) * (rd_val / MSF_liq)
        else:
            csr_val = 0.0
        csr_list.append(csr_val)

        # Factor of Safety against Liquefaction
        if csr_val > 0:
            fs = crr_val / csr_val
        else:
            fs = 99.0
        fs_liq_list.append(fs)

        if fs < 1.0 and z > 0:
            status_list.append("Liquefied")
        else:
            status_list.append("Non-Liquefied")

    # Determine maximum depth of full liquefaction (ru = 1)
    liq_depth_values = [z for z, st in zip(depths, status_list) if st == "Liquefied"]
    z_liq_max = max(liq_depth_values) if len(liq_depth_values) > 0 else 0.0

    idx_liq = np.where(depths == z_liq_max)[0]
    if len(idx_liq) > 0 and z_liq_max > 0:
        sig_eff_at_z_liq = sigma_v0_eff_list[idx_liq[0]]
    else:
        sig_eff_at_z_liq = 0.0

    # Extrapolate excess pore pressure ratio (r_u) profile
    ru_list = []
    ru_list = []
    for z, s_v0_eff, status_item in zip(depths, sigma_v0_eff_list, status_list):
        if z == 0:
            ru_list.append(1.0)
        elif z <= z_liq_max or status_item == "Liquefied":
            ru_list.append(1.0)
        else:
            ru_val = min(sig_eff_at_z_liq / s_v0_eff, 1.0) if s_v0_eff > 0 else 0.0
            ru_list.append(ru_val)

    # Summary Metrics
    l_col1, l_col2, l_col3, l_col4 = st.columns(4)
    l_col1.metric("Earthquake Magnitude ($M_w$)", f"{M_w:.1f}")
    l_col2.metric("PGA ($a_g$)", f"{a_g:.2f} g")
    l_col3.metric("Magnitude Scaling Factor (MSF)", f"{MSF_liq:.3f}")
    l_col4.metric("Depth of Full Liquefaction ($r_u = 1$)", f"{z_liq_max:.1f} m")

    st.markdown("---")
    
    # Visual Plots
    c_p1, c_p2 = st.columns(2)

    with c_p1:
        st.markdown("##### 📈 Liquefaction Evaluation (CSR vs q_c / p_a Curve)")
        qcn_curve = np.linspace(10, 250, 300)
        crr_curve = np.exp((qcn_curve / 540.0) + (qcn_curve / 67.0)**2 - (qcn_curve / 80.0)**3 + (qcn_curve / 114.0)**4 - 3.0)

        fig_liq1, ax_l1 = plt.subplots(figsize=(6, 5))
        ax_l1.plot(qcn_curve, crr_curve, 'k-', linewidth=2, label="CRR Boundary (CRR = CSR)")

        # Scatter plot for CPT soil layers
        mask_liq = np.array(status_list) == "Liquefied"
        mask_non_liq = np.array(status_list) == "Non-Liquefied"

        ax_l1.scatter(np.array(qcn_list)[mask_liq], np.array(csr_list)[mask_liq], 
                      color='red', marker='o', s=50, label="Liquefied Zone (FS < 1.0)")
        ax_l1.scatter(np.array(qcn_list)[mask_non_liq], np.array(csr_list)[mask_non_liq], 
                      color='blue', marker='^', s=50, label="Non-Liquefied Zone (FS ≥ 1.0)")

        ax_l1.text(50, 0.35, "LIQUEFIABLE", fontsize=11, fontweight='bold', color='red')
        ax_l1.text(170, 0.22, "NON-LIQUEFIABLE", fontsize=11, fontweight='bold', color='green')

        ax_l1.set_xlabel("Normalised Cone Resistance, $q_c / p_a$", fontsize=11)
        ax_l1.set_ylabel("Cyclic Stress Ratio, CSR", fontsize=11)
        ax_l1.set_xlim(0, 260)
        ax_l1.set_ylim(0, 0.6)
        ax_l1.grid(True, linestyle="--", alpha=0.6)
        ax_l1.legend(loc="upper left")
        st.pyplot(fig_liq1)

    with c_p2:
        st.markdown("##### 📉 Extrapolated Excess Pore Pressure Ratio ($r_u$) Profile")
        fig_liq2, ax_l2 = plt.subplots(figsize=(5, 5))
        ax_l2.plot(ru_list, depths, 'o-k', linewidth=2, markersize=5)
        ax_l2.axhline(y=z_liq_max, color='r', linestyle='--', label=f"Full Liquefaction Depth ({z_liq_max}m)")
        
        ax_l2.set_xlabel("Excess Pore Pressure Ratio, $r_u$", fontsize=11)
        ax_l2.set_ylabel("Depth (m)", fontsize=11)
        ax_l2.set_xlim(0, 1.1)
        ax_l2.set_ylim(max(depths), 0)  # Inverted depth axis
        ax_l2.grid(True, linestyle="--", alpha=0.6)
        ax_l2.legend(loc="lower left")
        st.pyplot(fig_liq2)

    st.markdown("---")
    st.subheader("📋 Step 7 Calculation Results Table")
    
    liq_df = pd.DataFrame({
        "Depth z (m)": depths,
        "qc (MPa)": qc_values,
        "qcn (qc/pa)": np.round(qcn_list, 1),
        "σ_v0 Total (kPa)": np.round(sigma_v0_total_list, 1),
        "σ'v0 Effective (kPa)": np.round(sigma_v0_eff_list, 1),
        "rd": np.round(rd_list, 3),
        "CSR": np.round(csr_list, 3),
        "CRR": np.round(crr_list, 3),
        "FS_liq": np.round(fs_liq_list, 2),
        "Status": status_list,
        "ru": np.round(ru_list, 3)
    })
    st.dataframe(liq_df, use_container_width=True)

    with st.expander("📖 **Step-by-Step Calculation Details (Liquefaction Potential - Example 7)**", expanded=False):
        st.write(r"#### 1. Cyclic Resistance Ratio (CRR)")
        st.latex(r"CRR = \exp \left[ \frac{q_{cn}}{540} + \left(\frac{q_{cn}}{67}\right)^2 - \left(\frac{q_{cn}}{80}\right)^3 + \left(\frac{q_{cn}}{114}\right)^4 - 3 \right]")
        st.write(f"* Normalised Cone Resistance: $q_{{cn}} = \\frac{{q_c}}{{p_a}}$ where $p_a = 100\\text{{ kPa}}$")

        st.write(r"#### 2. Cyclic Stress Ratio (CSR)")
        st.latex(r"CSR = 0.65 \left( \frac{a_{\max} \cdot \sigma_{v0}}{\sigma'_{v0}} \right) \frac{r_d}{MSF}")
        
        st.write(r"#### 3. Depth Reduction Factor ($r_d$) & Magnitude Scaling Factor (MSF)")
        st.latex(r"r_d = \exp(\alpha + \beta M)")
        st.latex(r"\alpha = -1.012 - 1.126 \sin\left(\frac{z}{11.73} + 5.133\right)")
        st.latex(r"\beta = 0.106 + 0.118 \sin\left(\frac{z}{11.28} + 5.142\right)")
        st.latex(rf"MSF = 6.9 \exp\left(-\frac{{M}}{{4}}\right) - 0.058 = {MSF_liq:.3f}")

        st.write(r"#### 4. Excess Pore Pressure Ratio ($r_u$) Extrapolation")
        st.write(f"* For $z \\le {z_liq_max}\\text{{ m}}$ (Liquefied Zone): $r_u = 1.0$")
        st.write(rf"* For $z > {z_liq_max}\text{{ m}}$ (Dense Non-Liquefied Zone): Excess pore pressure is assumed constant below liquefiable boundary $\Delta u(z) = \sigma'_{{v0}}({z_liq_max}\text{{m}}) = {sig_eff_at_z_liq:.1f}\text{{ kPa}}$.")
        st.latex(r"r_u(z) = \frac{\sigma'_{v0}(z_{\text{liq}})}{\sigma'_{v0}(z)}")

# =========================================================
# STEP 8: PILE SIZING BASED ON LIQUEFACTION CONSIDERATIONS (EX 8)
# =========================================================
with tab_ex8:
    st.subheader("6.4.2 Example 8: Pile Sizing based on Liquefaction Considerations")
    st.caption("Axial Failure Modes (ULS & SLS) under Liquefaction Conditions")

    col_s8_1, col_s8_2 = st.columns([1.1, 1.0])

    with col_s8_1:
        st.markdown("##### ⚙️ Input Parameters for Sizing")
        
        # Check if variables from previous steps exist, else fallback to defaults
        z_liq_default = float(z_liq_max) if 'z_liq_max' in locals() and z_liq_max > 0 else 8.0
        
        z_L_input = st.number_input(
            "Liquefaction Depth, $z_L$ (m)", 
            value=z_liq_default, 
            step=0.5,
            help="Step 7 မှ တွက်ချက်ရရှိထားသော Liquefaction ဖြစ်ပေါ်သည့် အနက်"
        )
        
        phi_deg = st.number_input("Soil Friction Angle, $\\phi$ (°)", value=30.0, step=1.0)
        alpha_ult = st.number_input("Base Capacity Ratio, $\\alpha_{ult} = Q_b / Q_u$", value=0.10, step=0.01)
        fos_static_target = st.number_input("Target Static FOS", value=2.0, step=0.1)

    with col_s8_2:
        st.markdown("##### 📐 Preliminary Bounds (ULS & Instability Check)")
        
        # Calculate design bounds based on Fig 6.8 & Ex 8 criteria
        L_p_min_chart = 11.0
        L_p_max_chart = 21.0

        # Fixed: Removed unsafe_allow_html=True from st.info
        st.info(
            f"**Design Range from ULS Instability Chart (Fig 6.8):**\n\n"
            f"**{L_p_min_chart:.1f} m < $L_p$ < {L_p_max_chart:.1f} m**"
        )

    st.markdown("---")
    st.subheader("📊 FOS Criteria and Required Pile Count ($N$) Curves")

    # Generate range of Pile Lengths (L_p)
    L_p_array = np.linspace(1.0, 25.0, 200)
    
    # 1. Demand Curve r_u,base = z_L / L_p (Eq 6.68)
    r_u_base_array = np.clip(z_L_input / L_p_array, 0.0, 1.0)

    # 2. FOS for Liquefaction-induced Bearing Capacity Failure (Eq 6.69)
    phi_rad = np.radians(phi_deg)
    exp_bearing = (3.0 - np.sin(phi_rad)) / (3.0 * (1.0 + np.sin(phi_rad)))
    
    fos_bearing_array = []
    for ru in r_u_base_array:
        if ru >= 1.0:
            fos_b = 15.0  # Allow higher FOS for shallow liquefaction depth
        else:
            denom = alpha_ult * ((1.0 - ru) ** exp_bearing) - alpha_ult + 1.0
            fos_b = 1.0 / denom if denom > 0 else 15.0
        fos_bearing_array.append(fos_b)

    # 3. FOS for Liquefaction-induced Settlement (Eq 6.70)
    fos_settlement_array = 1.0 + 5.5 * (r_u_base_array ** 3.5)

    # 4. Static FOS criterion
    fos_static_array = np.full_like(L_p_array, fos_static_target)

    # Maximum governing FOS at each depth
    fos_max_array = np.maximum.reduce([fos_static_array, fos_bearing_array, fos_settlement_array])

    # Reference values with safe fallback
    Q_u_ref = Q_u_mtd if 'Q_u_mtd' in locals() and Q_u_mtd > 0 else 5000.0
    P_axial_ref = P_axial if 'P_axial' in locals() and P_axial > 0 else 2000.0

    # Calculate depth-dependent single pile capacity Qu(Lp)
    # Skin friction reduces as Lp gets shorter
    Q_u_base = Q_u_ref * alpha_ult
    Q_u_skin_max = Q_u_ref * (1.0 - alpha_ult)
    
    Q_u_lp_array = []
    for lp in L_p_array:
        # Effective capacity drops inside liquefiable depth (Lp <= z_L)
        if lp <= z_L_input:
            qu = Q_u_base + Q_u_skin_max * (lp / 25.0) * 0.15
        else:
            qu = Q_u_base + Q_u_skin_max * (lp / 25.0)
        Q_u_lp_array.append(max(qu, 100.0))

    Q_u_lp_array = np.array(Q_u_lp_array)

    # Minimum number of piles required N = (P_total * FOS_max) / Q_u(Lp)
    N_required_array = (P_axial_ref * 1000.0 * fos_max_array) / Q_u_lp_array

    # Plots (Fig 6.9 representation)
    p_col1, p_col2 = st.columns(2)

    with p_col1:
        fig_fos, ax_f = plt.subplots(figsize=(6, 5))
        ax_f.plot(fos_static_array, L_p_array, 'o-', color='black', markevery=10, markersize=4, label=f"Static, FOS = {fos_static_target:.0f}")
        ax_f.plot(fos_bearing_array, L_p_array, 's-', color='navy', markevery=10, markersize=4, label="Bearing capacity")
        ax_f.plot(fos_settlement_array, L_p_array, '^-', color='darkgreen', markevery=10, markersize=4, label="Settlement")
        
        ax_f.axhline(y=z_L_input, color='red', linestyle='--', alpha=0.7, label=f"Liquefaction Depth ({z_L_input}m)")
        ax_f.axhline(y=12.0, color='orange', linestyle=':', alpha=0.8, label="Static Dominance Bound (12m)")

        ax_f.set_xlabel("Minimum FOS", fontsize=11, fontweight='bold')
        ax_f.set_ylabel("Pile Length $L_p$ (m)", fontsize=11, fontweight='bold')
        ax_f.set_xlim(0, 10)
        ax_f.set_ylim(25, 0)  # Inverted depth axis
        ax_f.grid(True, linestyle="--", alpha=0.6)
        ax_f.legend(loc="lower right")
        st.pyplot(fig_fos)

    with p_col2:
        fig_n, ax_n = plt.subplots(figsize=(6, 5))
        ax_n.semilogx(N_required_array, L_p_array, 'o-r', markevery=10, markersize=4, linewidth=2, label="Req. Pile Group Size N")
        
        ax_n.axhline(y=L_p_min_chart, color='gray', linestyle='--')
        ax_n.axhline(y=L_p_max_chart, color='gray', linestyle='--')
        ax_n.axvspan(1, 1000, ymin=1 - (L_p_max_chart/25.0), ymax=1 - (L_p_min_chart/25.0), color='green', alpha=0.1, label="Suitable ULS Range")

        ax_n.set_xlabel("Minimum number of piles N", fontsize=11, fontweight='bold')
        ax_n.set_ylabel("Pile Length $L_p$ (m)", fontsize=11, fontweight='bold')
        ax_n.set_xlim(1, 1000)
        ax_n.set_ylim(25, 0)  # Inverted depth axis

        # X-axis tick labels ကို 1, 10, 100, 1000 အဖြစ် ပြောင်းလဲခြင်း
        ax_n.set_xticks([1, 10, 100, 1000])
        ax_n.set_xticklabels(['1', '10', '100', '1000'])

        ax_n.grid(True, which="both", linestyle="--", alpha=0.6)
        ax_n.legend(loc="lower right")
        st.pyplot(fig_n)

    st.markdown("---")
    st.subheader("📋 Table 6.4: Summary of Final Pile Group Design")

    D_0_val = f"{D_0:.2f} m" if 'D_0' in locals() else "1.00 m"
    EI_val = f"{E_I/1e6:.0f} MN·m²" if 'E_I' in locals() else "2000 MN·m²"

    design_summary_df = pd.DataFrame({
        "Parameter": ["Pile Outer Diameter ($D_0$)", "Flexural Rigidity ($EI$)", "Yield Moment Capacity", "Suitable Pile Length Range ($L_p$)", "Min. Piles in Group ($N$)"],
        "Value": [D_0_val, EI_val, "1800 kNm", f"12 m < L_p < {L_p_max_chart:.0f} m", "N ≥ 4 (for all L_p > 12m)"],
        "Design Condition / Constraint": ["Selected Section", "Section Modulus", "Structural Limit", "Static Condition Dominates & Avoids Instability", "Satisfies Axial & Liquefaction Criteria"]
    })
    st.dataframe(design_summary_df, use_container_width=True)

    with st.expander("📖 **Step-by-Step Calculation Details (Example 8: Pile Sizing)**", expanded=False):
        st.write(r"#### 1. Demand Curve Equation")
        st.latex(rf"r_{{u,base}} = \frac{{z_L}}{{L_p}} = \frac{{{z_L_input:.1f}}}{{L_p}}")

        st.write(r"#### 2. Liquefaction-Induced Bearing Failure Criterion")
        st.latex(r"FOS \ge \frac{1}{\alpha_{ult} \left(1 - r_{u,base}\right)^{\frac{3-\sin\phi}{3(1+\sin\phi)}} - \alpha_{ult} + 1}")

        st.write(r"#### 3. Liquefaction-Induced Settlement Criterion (>0.1 D₀)")
        st.latex(r"FOS \ge 1 + 5.5 \left( r_{u,base} \right)^{3.5}")

        st.write(r"#### 4. Minimum Required Number of Piles ($N$)")
        st.latex(r"N = \frac{P_{total} \cdot FOS_{max}}{P_{ult}}")

        st.markdown("""
        **Summary of Design Regions (Fig 6.9):**
        * **$L_p < 8\\text{m}$:** Liquefiable layer ထဲတွင် pile တည်ရှိသဖြင့် Bearing failure စိုးမိုးပြီး FOS အလွန်မြင့်ရန် လိုအပ်သဖြင့် မသင့်တော်ပါ။
        * **$8\\text{m} < L_p < 12\\text{m}$:** Liquefaction-induced settlement စိုးမိုးပါသည်။ Pile အရေအတွက် ပိုမိုလိုအပ်ပါသည်။
        * **$L_p > 12\\text{m}$:** Static FOS condition ($FOS=2$) က စိုးမိုးပြီး Liquefaction ကြောင့် axial စွမ်းဆောင်ရည် ထိခိုက်မှု မရှိတော့ပါ။
        * **$L_p > 21\\text{m}$:** Instability (Buckling/ULS) ကြောင့် Pile Length ကို $21\\text{m}$ ထက် မပိုသင့်ပါ။
        """)

# =========================================================
# STEP 9: INERTIAL RESPONSE IN LEVEL LIQUEFIED GROUND (EX 9)
# =========================================================
with tab_ex9:
    st.subheader("6.4.3 Example 9: Inertial Response in Level Liquefied Ground")
    st.caption("Assumes zero stiffness ($E_{sD} \\approx 0$) in liquefied layer and Davisson (1970) fixity depth in dense sand.")

    col_s9_1, col_s9_2 = st.columns([1.1, 1.0])

    with col_s9_1:
        st.markdown("##### ⚙️ Input Parameters for Liquefied Ground")
        
        L_liq_default = float(z_liq_max) if 'z_liq_max' in locals() and z_liq_max > 0 else 8.0
        L_liq = st.number_input(
            "Liquefied Layer Thickness / Free-standing Length, $L_{p,layer1}$ (m)",
            value=L_liq_default, step=0.5
        )
        
        k_dense = st.number_input(
            "Subgrade Modulus of Underlying Dense Layer, $k$ (kN/m³)",
            value=9000.0, step=500.0
        )

        spec_factor_ex9 = st.number_input(
            "Spectral Acceleration Ratio ($S_{da} / a_g$)",
            value=0.90, step=0.05,
            help="Eurocode 8 design spectrum value for lengthened natural period (Damping ≈ 20%)"
        )

    with col_s9_2:
        st.markdown("##### 🏗️ System & Group Parameters (from Step 5)")
        n_piles_ex9 = n_piles if 'n_piles' in locals() else 4
        m_super_ex9 = m_super if 'm_super' in locals() else 940.0
        a_g_ex9 = a_g if 'a_g' in locals() else 0.2

        st.write(f"* **Number of Piles ($N_{{group}}$):** {n_piles_ex9}")
        st.write(f"* **Superstructure Mass ($m$):** {m_super_ex9:.0f} tons")
        st.write(f"* **Peak Ground Acceleration ($a_g$):** {a_g_ex9:.2f} g")

    # Calculations
    E_I_val = E_I if 'E_I' in locals() else (E_pile * 1e9 * I_p)
    T_u_ex9 = ((E_I_val) / (k_dense * 1e3)) ** 0.2

    # Fixity depth after Davisson (1970)
    if L_liq < T_u_ex9:
        L_f = 2.2 * T_u_ex9
    else:
        L_f = 1.8 * T_u_ex9

    L_total_eff = L_liq + L_f
    K_h_single_ex9 = (12 * E_I_val) / (L_total_eff ** 3)  # N/m
    K_h_single_MN = K_h_single_ex9 / 1e6                  # MN/m

    K_h_group_ex9 = n_piles_ex9 * K_h_single_ex9           # N/m
    mass_kg_ex9 = m_super_ex9 * 1000.0

    f_p_ex9 = (1.0 / (2 * np.pi)) * np.sqrt(K_h_group_ex9 / mass_kg_ex9)
    T_p_ex9 = 1.0 / f_p_ex9 if f_p_ex9 > 0 else 0.0

    a_resp_ex9 = spec_factor_ex9 * a_g_ex9 * 9.81         # m/s²
    H_inertial_ex9 = (mass_kg_ex9 * a_resp_ex9) / 1e6     # MN
    delta_h_ex9 = H_inertial_ex9 / (K_h_group_ex9 / 1e6)  # m

    st.markdown("---")
    st.markdown("##### 📊 Calculation Results Summary")

    r9_1, r9_2, r9_3, r9_4 = st.columns(4)
    r9_1.metric("Characteristic Length ($T_u$)", f"{T_u_ex9:.2f} m")
    r9_2.metric("Fixity Depth ($L_f$)", f"{L_f:.2f} m")
    r9_3.metric("Single Pile Stiffness ($K_{h\_eq}$)", f"{K_h_single_MN:.2f} MN/m")
    r9_4.metric("Natural Period ($T_p$)", f"{T_p_ex9:.2f} s")

    r9_5, r9_6, r9_7, r9_8 = st.columns(4)
    r9_5.metric("Response Acc. ($a_{resp}$)", f"{a_resp_ex9:.2f} m/s²")
    r9_6.metric("Inertial Load ($H$)", f"{H_inertial_ex9:.2f} MN")
    r9_7.metric("Peak Displacement ($\delta_h$)", f"{delta_h_ex9*1000:.1f} mm")
    r9_8.metric("Equivalent Free Length", f"{L_total_eff:.2f} m")

    st.markdown("---")
    st.markdown("##### 🔄 Comparison: Non-Liquefied Ground (Ex 5) vs. Liquefied Ground (Ex 9)")

    if 'K_h_eq_single' in locals() and 'delta_h_m' in locals():
        k_orig = K_h_eq_single / 1e6
        disp_orig = delta_h_m * 1000.0
        tp_orig = T_p_group
        a_orig = a_response
    else:
        k_orig, disp_orig, tp_orig, a_orig = 10.0, 10.0, 0.5, 4.0

    comp_df = pd.DataFrame({
        "Response Parameter": [
            "Single Pile Lateral Stiffness (K_h)",
            "Foundation Natural Period (T_p)",
            "Response Acceleration (a_resp)",
            "Peak Horizontal Displacement (δ_h)"
        ],
        "Non-Liquefied Ground (Ex 5)": [
            f"{k_orig:.2f} MN/m",
            f"{tp_orig:.2f} s",
            f"{a_orig:.2f} m/s²",
            f"{disp_orig:.1f} mm"
        ],
        "Liquefied Ground (Ex 9)": [
            f"{K_h_single_MN:.2f} MN/m",
            f"{T_p_ex9:.2f} s",
            f"{a_resp_ex9:.2f} m/s²",
            f"{delta_h_ex9*1000:.1f} mm"
        ],
        "Impact of Liquefaction": [
            "Reduces the lateral stiffness substantially",
            "Lengthens the natural period of the foundation",
            "Reduces acceleration at top (superstructure input)",
            "Increases lateral response substantially"
        ]
    })
    st.dataframe(comp_df, use_container_width=True)

    with st.expander("📖 **Step-by-Step Calculation Details (Example 9: Inertial Response in Liquefied Ground)**", expanded=False):
        st.write(r"#### 1. Relative Pile-Soil Stiffness in Dense Sand ($T_u$)")
        st.latex(rf"T_u = \left( \frac{{E_p I_p}}{{k}} \right)^{{0.2}} = \left( \frac{{{E_I_val/1e6:.1f} \times 10^6}}{{{k_dense:.0f} \times 10^3}} \right)^{{0.2}} = {T_u_ex9:.2f} \text{{ m}}")

        st.write(r"#### 2. Fixity Depth within Dense Layer ($L_f$) - Davisson (1970)")
        st.write(f"* Since $L_{{liq}} = {L_liq:.1f}\\text{{m}} > T_u = {T_u_ex9:.2f}\\text{{m}}$, fixity depth is $1.8 T_u$:")
        st.latex(rf"L_f = 1.8 T_u = 1.8 \times {T_u_ex9:.2f} = {L_f:.2f} \text{{ m}} \quad (\approx 5D_0)")

        st.write(r"#### 3. Single Pile Equivalent Lateral Stiffness ($K_{h\_eq}$)")
        st.latex(rf"K_{{h\_eq}} = \frac{{12 E_p I_p}}{{\left( L_{{p,layer1}} + L_f \right)^3}} = \frac{{12 \times {E_I_val/1e6:.1f} \times 10^6}}{{\left( {L_liq:.1f} + {L_f:.2f} \right)^3}} = {K_h_single_MN:.2f} \text{{ MN/m}}")

        st.write(r"#### 4. Natural Frequency ($f_p$) and Natural Period ($T_p$)")
        kh_str = f"{K_h_single_MN:.2f}"
        mass_str = f"{mass_kg_ex9:.0f}"
        fp_str = f"{f_p_ex9:.2f}"

        latex_code = (
            r"f_p = \frac{1}{2\pi} \sqrt{\frac{N_{group} \cdot K_{h\_eq}}{m}} = "
            r"\frac{1}{2\pi} \sqrt{\frac{%s \times %s \times 10^6}{%s}} = "
            r"%s \text{ Hz}" % (n_piles_ex9, kh_str, mass_str, fp_str)
        )

        st.latex(latex_code)
        st.latex(r"T_p = \frac{1}{f_p} = %.2f \text{ s}" % T_p_ex9)

        st.write(r"#### 5. Horizontal Inertial Load ($H$) & Peak Displacement ($\delta_h$)")

        st.latex(
            r"a_{response} = \left( \frac{S_{da}}{a_g} \right) \cdot a_g = %s \times %.2f \text{g} = %.2f \text{ m/s}^2"
            % (spec_factor_ex9, a_g_ex9, a_resp_ex9)
        )

        st.latex(
            r"H = m \cdot a_{response} = %.0f \times 10^3 \times %.2f = %.2f \text{ MN}"
            % (m_super_ex9, a_resp_ex9, H_inertial_ex9)
        )

        st.latex(
            r"\delta_h = \frac{H}{N_{group} \cdot K_{h\_eq}} = \frac{%.2f}{%s \times %.2f} = %.3f \text{ m} \quad (%.1f \text{ mm})"
            % (H_inertial_ex9, n_piles_ex9, K_h_single_MN, delta_h_ex9, delta_h_ex9 * 1000)
        )

# ==========================================
# STEP 10: LATERAL SPREADING ANALYSIS (EX 10)
# ==========================================
with tab_ex10:
    st.header("Step 10: Pile Group in Soil Subject to Lateral Spreading")
    st.markdown("""
    This step evaluates the foundation performance under **Lateral Spreading** on sloping ground ($3^\circ$) 
    using the **Limiting Lateral Earth Pressure Approach** (Example 10 / Cubrinovski et al. method).
    """)

    # Ensure global variables have fallback values if not defined in sidebar
    if 'V_total' not in globals():
        V_total = 9400.0  # kN (Default Total Superstructure Load)
    if 'L_pile' not in globals():
        L_pile = 20.0     # m (Default Pile Length)

    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        q_lat = st.number_input("Limiting Lateral Pressure, q_lat (kPa)", value=20.0)
        slope_deg = st.number_input("Ground Slope Angle (°)", value=3.0)
    with col_in2:
        B_cap = st.number_input("Pile Cap Width, B (m)", value=6.0)
        t_cap = st.number_input("Pile Cap Thickness, t (m)", value=1.5)
    with col_in3:
        L_eff_sp = st.number_input("Effective Pile Length above Fixity, L (m)", value=11.83)
        H_peak = st.number_input("Peak Inertial Force from Structure, H (kN)", value=1690.0)

    f_delta = st.number_input("Displacement Magnification Factor, f_Δ", value=7.2)
    f_pl = 0.0695 # Soil pressure moment factor

    st.subheader("1. Design Cases Comparison & Mitigation Options")
    
    # Calculation Function (Updated with V_tot argument)
    def calc_lateral_spreading(N_piles, D_p, EI_val, My_val, H_force, V_tot):
        F_cap = q_lat * B_cap * t_cap # kN
        p_L = q_lat * D_p # kN/m
        P_axial = V_tot / N_piles # kN per pile
        
        mu = np.sqrt(P_axial / EI_val)
        mu_L = mu * L_eff_sp
        
        # Lateral displacement Eq 6.78
        num = (2 * (F_cap + H_force) + N_piles * p_L * D_p * L_eff_sp) * L_eff_sp
        den = 2 * N_piles * P_axial * (2 * f_delta - 1)
        delta_h = num / den # in meters
        
        # Yield displacement Eq 6.84
        delta_yield = (My_val - (p_L * D_p / (mu**2)) * f_pl) / (P_axial * f_delta)
        
        return F_cap, p_L, P_axial, delta_h, delta_yield

    # Structural Parameters
    EI_075, My_075 = 398000.0, 1800.0  # D = 0.75m
    EI_100, My_100 = 1257000.0, 4260.0 # D = 1.00m

    # Calculate for Case 1 (Original 2x2, D=0.75m)
    _, _, P1, d_res1, d_y1 = calc_lateral_spreading(4, 0.75, EI_075, My_075, 0.0, V_total)
    _, _, _, d_peak1, _ = calc_lateral_spreading(4, 0.75, EI_075, My_075, H_peak, V_total)

    # Calculate for Case 2 (3x3, D=0.75m)
    _, _, P2, d_res2, d_y2 = calc_lateral_spreading(9, 0.75, EI_075, My_075, 0.0, V_total)
    _, _, _, d_peak2, _ = calc_lateral_spreading(9, 0.75, EI_075, My_075, H_peak, V_total)

    # Calculate for Case 3 (2x2, D=1.00m)
    _, _, P3, d_res3, d_y3 = calc_lateral_spreading(4, 1.00, EI_100, My_100, 0.0, V_total)
    _, _, _, d_peak3, _ = calc_lateral_spreading(4, 1.00, EI_100, My_100, H_peak, V_total)

    # Steel volume calculation
    vol_steel_1 = 4 * np.pi * (0.75 * 0.016 - 0.016**2) * L_pile
    vol_steel_2 = 9 * np.pi * (0.75 * 0.016 - 0.016**2) * L_pile
    vol_steel_3 = 4 * np.pi * (1.00 * 0.016 - 0.016**2) * L_pile

    # Summary Table Data
    summary_data = {
        "Metric": [
            "Group Configuration", "Number of Piles (N)", "Pile Diameter, D₀ (m)",
            "Axial Load / Pile (kN)", "Residual Disp. δ_res (mm)", "Peak Disp. δ_h (mm)",
            "Yield Disp. δ_yield (mm)", "Ratio (δ_h / δ_yield)", "Steel Volume (m³)",
            "Pile Cap Size (m)", "Design Status"
        ],
        "Design Ex. 9 (Original)": [
            "2 × 2", "4", "0.75", f"{P1:.0f}",
            f"{d_res1*1000:.1f}", f"{d_peak1*1000:.1f}", f"{d_y1*1000:.1f}",
            f"{d_peak1/d_y1:.2f}", f"{vol_steel_1:.2f}", "6.00",
            "❌ Unsuitable (Yields in spreading soil)"
        ],
        "Method 2 (More Piles)": [
            "3 × 3", "9", "0.75", f"{P2:.0f}",
            f"{d_res2*1000:.1f}", f"{d_peak2*1000:.1f}", f"{d_y2*1000:.1f}",
            f"{d_peak2/d_y2:.2f}", f"{vol_steel_2:.2f}", "9.75",
            "⚠️ Suitable, but costly"
        ],
        "Method 3 (Larger Piles)": [
            "2 × 2", "4", "1.00", f"{P3:.0f}",
            f"{d_res3*1000:.1f}", f"{d_peak3*1000:.1f}", f"{d_y3*1000:.1f}",
            f"{d_peak3/d_y3:.2f}", f"{vol_steel_3:.2f}", "8.00",
            "✅ Suitable & Optimal (Best performance)"
        ]
    }
    
    st.table(pd.DataFrame(summary_data))

    # Parametric Plots
    st.subheader("2. Lateral Displacement vs. Number of Piles (Parametric Curves)")
    
    N_range = np.arange(4, 11)
    d_h_075, d_h_100 = [], []

    for n in N_range:
        _, _, _, dh_75, _ = calc_lateral_spreading(n, 0.75, EI_075, My_075, H_peak, V_total)
        _, _, _, dh_100, _ = calc_lateral_spreading(n, 1.00, EI_100, My_100, H_peak, V_total)
        d_h_075.append(dh_75 * 1000)
        d_h_100.append(dh_100 * 1000)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(N_range, d_h_075, 'k-o', label=r'Lateral Displacement $\delta_h$')
    ax1.axhline(d_y1 * 1000, color='r', linestyle='--', label=r'Yield Limit $\delta_{yield}$')
    ax1.set_xlabel("Number of Piles in Group, N")
    ax1.set_ylabel("Lateral Displacement, δ (mm)")
    ax1.set_title("D = 0.75m Tubular Steel Piles")
    ax1.set_ylim(0, 250)
    ax1.grid(True, linestyle=':')
    ax1.legend()

    ax2.plot(N_range, d_h_100, 'k-o', label=r'Lateral Displacement $\delta_h$')
    ax2.axhline(d_y3 * 1000, color='r', linestyle='--', label=r'Yield Limit $\delta_{yield}$')
    ax2.set_xlabel("Number of Piles in Group, N")
    ax2.set_ylabel("Lateral Displacement, δ (mm)")
    ax2.set_title("D = 1.00m Tubular Steel Piles")
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle=':')
    ax2.legend()

    st.pyplot(fig2)

