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

# Soil Type Multi-Parameter Correlation Rule
SOIL_DATABASE = {
    "NC Soft Clay": {"gamma": 16.0, "e_default": 1.10, "phi": 22.0},
    "OC Clay": {"gamma": 18.0, "e_default": 0.70, "phi": 28.0},
    "Loose / Silty Sand": {"gamma": 17.0, "e_default": 0.90, "phi": 30.0},
    "Medium Dense Sand": {"gamma": 18.5, "e_default": 0.75, "phi": 34.0},
    "Dense Sand / Granular": {"gamma": 20.0, "e_default": 0.60, "phi": 38.0},
    "Clean Sand": {"gamma": 19.0, "e_default": 0.65, "phi": 36.0}
}

def classify_soil_and_get_e(qc, selected_type):
    # Rule-Based Auto-Classification based on qc threshold & user selection
    if selected_type == "Auto Classify from qc":
        if qc < 1.5:
            return "NC Soft Clay", 1.10, 16.0
        elif 1.5 <= qc < 5.0:
            return "Loose / Silty Sand", 0.90, 17.0
        elif 5.0 <= qc < 15.0:
            return "Medium Dense Sand", 0.75, 18.5
        elif 15.0 <= qc < 25.0:
            return "Clean Sand", 0.65, 19.0
        else:
            return "Dense Sand / Granular", 0.60, 20.0
    else:
        info = SOIL_DATABASE[selected_type]
        return selected_type, info["e_default"], info["gamma"]

# ---------------------------------------------------------
# Main Input Section
# ---------------------------------------------------------
with st.expander("⚙️ **Design Parameters & Soil Test Data (Click to Expand/Collapse)**", expanded=True):
    col_p1, col_p2, col_p3 = st.columns([1, 1.3, 1])
    
    with col_p1:
        st.subheader("1. Pile Properties")
        pile_type = st.selectbox("Select Pile Type", ["Steel Pipe Pile", "Bored Concrete Pile"])
        D_0 = st.number_input("Pile Diameter, $D_0$ (m)", value=0.75, step=0.05)
        L_p = st.number_input("Pile Length, $L_p$ (m)", value=20.0, step=1.0)
        P_axial = st.number_input("Axial Load (MN)", value=9.4, step=0.1)

    with col_p2:
        st.subheader("2. CPT Field Soil Test Input")
        st.caption("Depth (m) နှင့် CPT $q_c$ (MPa) တန်ဖိုးများသာ ထည့်သွင်းရန် လိုအပ်ပါသည်။")
        
        default_cpt_df = pd.DataFrame({
            "Depth (m)": np.arange(0, 21, 1),
            "qc (MPa)": [0, 0.74, 1.19, 1.81, 1.82, 2.41, 2.79, 3.25, 3.43, 12.85, 13.77, 16.03, 17.94, 19.07, 17.88, 24.94, 20.93, 23.71, 22.97, 26.59, 28.89]
        })
        
        edited_cpt_df = st.data_editor(
            default_cpt_df,
            height=200,
            num_rows="fixed",
            use_container_width=True
        )
        
        depths = edited_cpt_df["Depth (m)"].values
        qc_values = edited_cpt_df["qc (MPa)"].values

    with col_p3:
        st.subheader("3. Soil Type Mapping & Settings")
        soil_override = st.selectbox(
            "Soil Type Determination", 
            ["Auto Classify from qc", "NC Soft Clay", "OC Clay", "Loose / Silty Sand", "Medium Dense Sand", "Clean Sand", "Dense Sand / Granular"]
        )

        # Dynamic array calculation for Soil Classification, Void Ratio, and Overburden Stress
        classified_types = []
        e_values = []
        gamma_values = []
        
        for q in qc_values:
            s_type, e_val, g_val = classify_soil_and_get_e(q, soil_override)
            classified_types.append(s_type)
            e_values.append(e_val)
            gamma_values.append(g_val)

        # Automatic Effective Stress Sigma_v0 calculation (Cumulative integration)
        sigma_v0 = [0.0]
        for i in range(1, len(depths)):
            dz = depths[i] - depths[i-1]
            gamma_eff = max(gamma_values[i] - 9.81, 7.0)  # Submerged effective unit weight
            sigma_v0.append(sigma_v0[-1] + gamma_eff * dz)

        e_silty = e_values[4] if len(e_values) > 4 else 0.90
        st.info(f"💡 **Top Layer Auto-Classified as:** `{classified_types[4] if len(classified_types) > 4 else classified_types[0]}` ($e \\approx {e_silty:.2f}$)")

        soil_profile = st.selectbox("Soil Profile Type", ["Parabolic (Sand / Silty Sand)", "Linear (Soft Clay)", "Constant (OC Clay)"])
        a_g = st.number_input("PGA, $a_g$ (g)", value=0.2, step=0.05)
        M_w = st.number_input("Earthquake Magnitude, $M_w$", value=6.0, step=0.1)
        apply_msf = st.checkbox("Apply MSF to $\\tau_{max}$?", value=False)

if pile_type == "Steel Pipe Pile":
    t_wall = 0.012
    E_pile = 210.0
    delta_cv = 20.0
else:
    t_wall = D_0 / 2.0
    fc_prime = 30.0
    E_pile = (4700 * np.sqrt(fc_prime)) / 1000.0
    delta_cv = 26.25

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
# STEP 1: CPT METHOD
# =========================================================
with tab_ex2:
    st.subheader("6.2.2 CPT-Based Axial Pile Capacity")
    st.caption("Field Investigation Step: Direct calculation using continuous cone resistance ($q_c$) profile.")

    d_cone = 25.4
    qc_tip_book = qc_values[-1]
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

    st.subheader("📋 Soil Classification Log per Depth")
    summary_df = pd.DataFrame({
        "Depth (m)": depths,
        "qc (MPa)": qc_values,
        "Auto-Calculated σ'v0 (kPa)": np.round(sigma_v0, 2),
        "Classified Soil Type": classified_types,
        "Mapped Void Ratio (e)": e_values
    })
    st.dataframe(summary_df, use_container_width=True)

# =========================================================
# STEP 2: BROMS STATIC METHOD
# =========================================================
with tab_ex1:
    st.subheader("6.2.1 Preliminary Design under Static Loading (Broms 1966)")
    st.caption("Analytical Verification Step: Capacity calculation based on soil strength properties.")

    H_top = 8.0
    A_b = (np.pi / 4) * (D_0 ** 2)
    sigma_b_eff = sigma_v0[-1]
    N_q = 40  
    Q_b_ex1 = A_b * sigma_b_eff * (N_q - 1)

    if pile_type == "Steel Pipe Pile":
        K_s1, K_s2, delta_deg = 0.5, 1.0, 20.0
        st.info("ℹ️ **Broms (1966) Steel Parameters:** $\\delta_{cv} = 20^\\circ$, $K_{s1}=0.5$, $K_{s2}=1.0$")
    else:
        K_s1, K_s2, delta_deg = 1.0, 2.0, 0.75 * 35.0
        st.info("ℹ️ **Broms (1966) Concrete Parameters:** $\\delta_{cv} = 26.25^\\circ$, $K_{s1}=1.0$, $K_{s2}=2.0$")

    tan_delta = np.tan(np.radians(delta_deg))
    Q_s_layer1 = K_s1 * tan_delta * (0.5 * 1.27 * (H_top**2))
    Q_s_layer2 = K_s2 * tan_delta * (0.5 * 3.28 * (L_p**2 - H_top**2))
    
    Q_s_ex1 = np.pi * D_0 * (Q_s_layer1 + Q_s_layer2)
    Q_u_ex1 = Q_b_ex1 + Q_s_ex1
    N_required_ex1 = (P_axial * 1000) / Q_u_ex1

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

    if "Parabolic" in soil_profile:
        exponent = 0.22
    elif "Linear" in soil_profile:
        exponent = 0.20
    else:
        exponent = 0.25

    if pile_type == "Steel Pipe Pile":
        D_i = D_0 - (2 * t_wall)
        E_p_corrected = E_pile / ((D_0**4) / (D_0**4 - D_i**4))
        I_p = (np.pi / 64) * (D_0**4 - D_i**4)
    else:
        E_p_corrected = E_pile
        I_p = (np.pi / 64) * (D_0**4)

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
