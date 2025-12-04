# cosmic_expansion_simulator.py
# Copy this ENTIRE code and run with: streamlit run cosmic_expansion_simulator.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')  # For Streamlit compatibility
from io import BytesIO
import base64

# ======================
# PAGE CONFIGURATION
# ======================
st.set_page_config(
    page_title="Cosmic Expansion Simulator",
    page_icon="🌌",
    layout="wide"
)

# ======================
# CUSTOM STYLING
# ======================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7B8B9A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .fate-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .big-crunch { background-color: #FF6B6B; color: white; }
    .flat-universe { background-color: #4ECDC4; color: white; }
    .eternal-accel { background-color: #45B7D1; color: white; }
    .slider-label { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ======================
# TITLE SECTION
# ======================
st.markdown('<h1 class="main-header">🌌 Cosmic Expansion Simulator</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Explore the Fate of the Universe: Big Crunch vs Eternal Acceleration</p>', unsafe_allow_html=True)

st.write("""
**How it works:** The universe's fate depends on the balance between **matter (gravity)** and **dark energy (repulsion)**.
Move the sliders to see how different ratios change cosmic destiny.
""")

# ======================
# SIDEBAR - CONTROLS
# ======================
with st.sidebar:
    st.markdown("### 🎛️ Cosmic Parameters")
    st.markdown("---")
    
    st.markdown('<p class="slider-label">Matter Density Ωₘ</p>', unsafe_allow_html=True)
    omega_m = st.slider(
        "Normal + Dark Matter",
        min_value=0.0,
        max_value=1.5,
        value=0.31,
        step=0.01,
        help="Gravity effect - pulls universe together"
    )
    
    st.markdown('<p class="slider-label">Dark Energy Density ΩΛ</p>', unsafe_allow_html=True)
    omega_lambda = st.slider(
        "Dark Energy (Cosmological Constant)",
        min_value=0.0,
        max_value=1.5,
        value=0.69,
        step=0.01,
        help="Anti-gravity effect - pushes universe apart"
    )
    
    st.markdown('<p class="slider-label">Hubble Constant H₀ (km/s/Mpc)</p>', unsafe_allow_html=True)
    h0 = st.slider(
        "Current Expansion Rate",
        min_value=50.0,
        max_value=90.0,
        value=70.0,
        step=0.1,
        help="How fast universe expands TODAY"
    )
    
    st.markdown("---")
    
    # Time range control
    st.markdown("### ⏱️ Simulation Time")
    time_range = st.slider(
        "Billions of Years from Now",
        min_value=-10.0,
        max_value=50.0,
        value=[-5.0, 20.0],
        step=0.5,
        help="Negative = past, Positive = future"
    )
    
    st.markdown("---")
    
    # Real universe comparison
    st.markdown("### 🌍 Real Universe Parameters")
    if st.button("Load Our Universe"):
        st.session_state.omega_m = 0.31
        st.session_state.omega_lambda = 0.69
        st.session_state.h0 = 70.0
    
    st.caption("Our Universe: Ωₘ=0.31, ΩΛ=0.69, H₀≈70")

# ======================
# FRIEDMANN EQUATION
# ======================
def friedmann_equation(z, omega_m, omega_lambda, h0):
    """
    Friedmann equation for flat universe (Ω_k = 0)
    H(z) = H0 * sqrt(Ω_m*(1+z)^3 + Ω_Λ)
    """
    hz = h0 * np.sqrt(omega_m * (1 + z)**3 + omega_lambda)
    return hz

def scale_factor_over_time(t_Gyr, omega_m, omega_lambda, h0):
    """
    Calculate scale factor R(t) by integrating Friedmann equation
    t in billions of years, H0 in km/s/Mpc
    """
    # Convert H0 to Gyr^-1: 1/(H0) ≈ 14 Gyr for H0=70
    H0_Gyr = h0 * 0.001022  # Conversion factor
    
    # For numerical solution, use dimensionless time
    t_norm = t_Gyr * H0_Gyr
    
    # Solve differential equation: dR/dt = H0 * sqrt(Ω_m/R + Ω_Λ*R^2)
    # Using analytical solution for flat universe
    R = np.zeros_like(t_norm)
    
    for i, tn in enumerate(t_norm):
        # Numerical integration for general case
        if omega_lambda == 0:
            # Matter-only universe
            R[i] = (1.5 * omega_m**0.5 * tn + 1)**(2/3)
        elif omega_m == 0:
            # Dark-energy-only universe
            R[i] = np.exp(omega_lambda**0.5 * tn)
        else:
            # Mixed case - use approximate solution
            # This is simplified for demonstration
            a = omega_m / (1 - omega_m)
            R[i] = (a * np.sinh(1.5 * np.sqrt(omega_lambda) * tn))**(2/3)
    
    return R

# ======================
# DETERMINE COSMIC FATE
# ======================
def determine_fate(omega_m, omega_lambda):
    """Determine the ultimate fate of the universe"""
    if omega_m == 0 and omega_lambda == 0:
        return "Empty Universe", "Static"
    
    total = omega_m + omega_lambda
    
    if total > 1.0 and omega_m > omega_lambda:
        return "Big Crunch", "Gravity wins - Universe collapses"
    elif total < 1.0 or (total >= 1.0 and omega_lambda > omega_m):
        return "Eternal Acceleration", "Dark energy wins - Forever expanding"
    elif abs(total - 1.0) < 0.01 and abs(omega_m - 0.3) < 0.1:
        return "Flat Universe", "Perfect balance - Coasting expansion"
    else:
        return "Critical Expansion", "Borderline case"

# ======================
# CREATE ANIMATION
# ======================
def create_animation(omega_m, omega_lambda, h0, time_range):
    """Create animated expansion plot"""
    # Time array (billions of years)
    t = np.linspace(time_range[0], time_range[1], 200)
    current_time_idx = np.argmin(np.abs(t))  # Index for "now" (t=0)
    
    # Calculate scale factor
    R = scale_factor_over_time(t, omega_m, omega_lambda, h0)
    
    # Normalize to R=1 at present
    if len(R) > current_time_idx:
        R_normalized = R / R[current_time_idx]
    else:
        R_normalized = R
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Expansion History
    ax1.plot(t, R_normalized, 'b-', linewidth=3, alpha=0.7, label=f'Ωₘ={omega_m:.2f}, ΩΛ={omega_lambda:.2f}')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Now (t=0)')
    ax1.axhline(y=1, color='gray', linestyle=':', alpha=0.3)
    
    # Mark current point
    if len(t) > current_time_idx:
        ax1.plot(t[current_time_idx], R_normalized[current_time_idx], 'ro', markersize=10, label='Present')
    
    ax1.set_xlabel('Time (Billions of Years from Now)', fontsize=12)
    ax1.set_ylabel('Scale Factor R(t) / R(today)', fontsize=12)
    ax1.set_title('Cosmic Expansion History', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Highlight region based on fate
    fate, _ = determine_fate(omega_m, omega_lambda)
    if "Crunch" in fate:
        ax1.fill_between(t[t<0], 0, R_normalized[t<0], alpha=0.2, color='red')
        ax1.text(-8, 0.8, 'Past', fontsize=10, color='red')
    elif "Acceleration" in fate:
        ax1.fill_between(t[t>0], 1, R_normalized[t>0], alpha=0.2, color='green')
        ax1.text(10, 2.5, 'Future Acceleration', fontsize=10, color='green')
    
    # Plot 2: Phase Space Diagram
    omega_m_range = np.linspace(0, 1.5, 50)
    omega_lambda_range = np.linspace(0, 1.5, 50)
    
    # Create grid for fate regions
    for om in np.linspace(0, 1.5, 20):
        for ol in np.linspace(0, 1.5, 20):
            fate, _ = determine_fate(om, ol)
            if "Crunch" in fate:
                ax2.plot(om, ol, 'r.', markersize=8, alpha=0.3)
            elif "Acceleration" in fate:
                ax2.plot(om, ol, 'g.', markersize=8, alpha=0.3)
            else:
                ax2.plot(om, ol, 'b.', markersize=8, alpha=0.3)
    
    # Plot current point
    ax2.plot(omega_m, omega_lambda, 'yellow', marker='*', markersize=20, 
             markeredgecolor='black', markeredgewidth=2, label='Your Universe')
    
    # Plot real universe
    ax2.plot(0.31, 0.69, 'w', marker='o', markersize=12, 
             markeredgecolor='black', label='Real Universe (Ωₘ=0.31, ΩΛ=0.69)')
    
    ax2.set_xlabel('Matter Density Ωₘ', fontsize=12)
    ax2.set_ylabel('Dark Energy ΩΛ', fontsize=12)
    ax2.set_title('Cosmic Fate Parameter Space', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    ax2.set_xlim([0, 1.5])
    ax2.set_ylim([0, 1.5])
    
    # Add fate regions text
    ax2.text(0.8, 0.3, 'Big Crunch\nRegion', fontsize=10, color='red', 
             ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    ax2.text(0.3, 0.8, 'Eternal\nAcceleration', fontsize=10, color='green',
             ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    ax2.text(0.5, 0.5, 'Flat/Coasting', fontsize=10, color='blue',
             ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

# ======================
# MAIN DISPLAY
# ======================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📈 Cosmic Expansion Visualization")
    
    # Generate plot
    fig = create_animation(omega_m, omega_lambda, h0, time_range)
    st.pyplot(fig)
    
    # Add explanation
    with st.expander("📖 What am I seeing?"):
        st.write("""
        **Left Plot (Expansion History):**
        - **X-axis:** Time (negative = past, positive = future)
        - **Y-axis:** Scale factor R(t) - relative size of universe
        - **Red dashed line:** Present day (t=0)
        - **Curve shape:**
          - Concave down (slowing): Gravity dominates
          - Concave up (accelerating): Dark energy dominates
        
        **Right Plot (Parameter Space):**
        - Each point represents a possible universe
        - **Red region:** Big Crunch (gravity wins)
        - **Green region:** Eternal Acceleration (dark energy wins)
        - **Blue region:** Flat/Coasting universe
        - **Yellow star:** Your current settings
        - **White circle:** Our actual universe
        """)

with col2:
    st.markdown("### 🎯 Cosmic Fate Prediction")
    
    # Determine fate
    fate, description = determine_fate(omega_m, omega_lambda)
    
    # Color-coded fate display
    if "Crunch" in fate:
        st.markdown(f'<div class="fate-box big-crunch">🌠 {fate}</div>', unsafe_allow_html=True)
    elif "Acceleration" in fate:
        st.markdown(f'<div class="fate-box eternal-accel">🚀 {fate}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="fate-box flat-universe">⚖️ {fate}</div>', unsafe_allow_html=True)
    
    st.write(f"**{description}**")
    
    # Key metrics
    st.markdown("#### 📊 Key Metrics")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Ωₘ (Matter)", f"{omega_m:.3f}", 
                  delta="Gravity" if omega_m > omega_lambda else None)
    with col_b:
        st.metric("ΩΛ (Dark Energy)", f"{omega_lambda:.3f}",
                  delta="Repulsion" if omega_lambda > omega_m else None)
    
    st.metric("Ωₘ/ΩΛ Ratio", f"{omega_m/omega_lambda:.3f}" if omega_lambda > 0 else "∞",
              delta="Gravity wins" if omega_m > omega_lambda else "Dark energy wins")
    
    st.metric("Hubble Constant", f"{h0:.1f} km/s/Mpc")
    
    # Expansion rate at different times
    st.markdown("#### ⏱️ Expansion Rates")
    
    redshifts = [0, 1, 5, 10]  # z=0 (now), z=1 (half size), etc.
    for z in redshifts:
        hz = friedmann_equation(z, omega_m, omega_lambda, h0)
        st.write(f"At z={z}: H(z) = {hz:.1f} km/s/Mpc")
    
    # What-if scenarios
    st.markdown("#### 🎮 Try These Scenarios")
    
    scenario_col1, scenario_col2 = st.columns(2)
    with scenario_col1:
        if st.button("Matter Dominated"):
            st.session_state.omega_m = 0.9
            st.session_state.omega_lambda = 0.1
    
    with scenario_col2:
        if st.button("Dark Energy Dominated"):
            st.session_state.omega_m = 0.1
            st.session_state.omega_lambda = 1.2

# ======================
# SCIENCE EXPLANATION
# ======================
st.markdown("---")
st.markdown("## 🔬 The Science Behind the Simulation")

col_sci1, col_sci2, col_sci3 = st.columns(3)

with col_sci1:
    st.markdown("### Friedmann Equation")
    st.latex(r"H(t)^2 = H_0^2 \left[ \Omega_m (1+z)^3 + \Omega_\Lambda \right]")
    st.write("Where:")
    st.write("- $H(t)$: Expansion rate at time t")
    st.write("- $H_0$: Today's expansion rate (Hubble constant)")
    st.write("- $\Omega_m$: Matter density parameter")
    st.write("- $\Omega_\Lambda$: Dark energy density parameter")
    st.write("- $z$: Redshift (measure of cosmic time)")

with col_sci2:
    st.markdown("### Our Universe (2025 Data)")
    st.write("**Planck Satellite + DESI 2024:**")
    st.write("- Ωₘ = 0.311 ± 0.006 (Matter)")
    st.write("- ΩΛ = 0.689 ± 0.006 (Dark Energy)")
    st.write("- H₀ = 67.4 ± 0.5 km/s/Mpc")
    st.write("")
    st.write("**Key Discovery:** Dark energy causes ACCELERATED expansion")

with col_sci3:
    st.markdown("### Three Possible Fates")
    st.write("""
    1. **Big Crunch** (Ωₘ > ΩΛ)
       - Gravity wins
       - Universe collapses
    
    2. **Flat/Coasting** (Ωₘ + ΩΛ = 1)
       - Perfect balance
       - Expansion slows to zero
    
    3. **Eternal Acceleration** (ΩΛ > Ωₘ)
       - Dark energy wins
       - **Our actual fate!**
    """)

# ======================
# PERSONAL REFLECTION SECTION
# ======================
st.markdown("---")
with st.expander("📝 Personal Reflection & Connection to My Poem", expanded=True):
    st.markdown("""
    ### From "Black shadows lurk..." to Cosmic Light
    
    In my poem *"Black shadows lurk in corners of my mind"*, I explore the human struggle with uncertainty and the unknown. 
    This cosmic simulator represents the scientific counterpart to that exploration — using mathematics and observation 
    to illuminate what was once complete darkness.
    
    **Parallel Journeys:**
    - **Poem:** Personal, internal darkness → seeking light
    - **Cosmology:** Cosmic darkness (95% unknown universe) → scientific illumination
    
    **The Connection:** Just as poetry helps us navigate internal uncertainties, tools like this simulator help humanity 
    navigate cosmic uncertainties. We've moved from mythical explanations to mathematical predictions, from fearing 
    the dark to measuring it with Planck precision.
    
    **My Insight:** The 2024 DESI data suggesting evolving dark energy resonates with my poem's theme of 
    *"shadows that shift and change"* — even our fundamental constants might not be constant. 
    This simulator lets me explore that possibility interactively.
    
    > *"We are stardust trying to understand the furnace that forged us,*
    > *using equations as our lanterns in the cosmic dark."*
    """)

# ======================
# FOOTER
# ======================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9rem;">
<p>Cosmic Expansion Simulator | Based on Friedmann Equations | ΛCDM Model</p>
<p>Data Sources: Planck 2018, DESI 2024, Hubble Constant Measurements</p>
<p>Created for Educational and Research Demonstration</p>
</div>
""", unsafe_allow_html=True)

# ======================
# SESSION STATE INITIALIZATION
# ======================
if 'omega_m' not in st.session_state:
    st.session_state.omega_m = omega_m
if 'omega_lambda' not in st.session_state:
    st.session_state.omega_lambda = omega_lambda
if 'h0' not in st.session_state:
    st.session_state.h0 = h0
