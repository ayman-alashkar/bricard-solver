"""
Bricard 6R Linkage Kinematic Solver
===================================
Interactive solver for the original general line-symmetric Bricard 6R linkage.

Based A Tensor Method for the Kinematical Analysis of the Line-Symmetric Bricard 6R Linkage.

Author: Ayman Alashkar
Affiliation: Mechanics and Materials Unit, OIST
Date: December 2025
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Bricard 6R Kinematic Solver",
    page_icon="⚙️",
    layout="wide"
)

# Title and description
st.title("⚙️ Bricard 6R Linkage Kinematic Solver")
st.markdown("""
Configure the Bricard 6R parameters. Due to symmetry ($i = i+3$), you only need to define the first 3 links.
""")

# ============================================================================
# Core Mathematical Functions
# ============================================================================

def compute_coefficients(a1, a2, a3, alpha1, alpha2, alpha3, R1, R2, R3):
    """
    Compute the closure equation coefficients.
    
    Parameters are for links 1, 2, 3 (links 4, 5, 6 are symmetric).
    Angles alpha should be in radians.
    """
    # Shorthand for trig functions
    sa1, ca1 = np.sin(alpha1), np.cos(alpha1)
    sa2, ca2 = np.sin(alpha2), np.cos(alpha2)
    sa3, ca3 = np.sin(alpha3), np.cos(alpha3)
    
    # Coefficients for theta2 equation
    A2 = a1 * sa3 + a3 * sa1 * ca2
    B2 = -(R2 * sa1 * sa3 + R3 * sa1 * ca2 * sa3)
    C2 = a1 * sa2 + a2 * ca3 * sa1
    D2 = -(R1 * sa1 * sa2 + R3 * sa1 * sa2 * ca3)
    E2 = R3 * sa2 * sa3
    F2 = a3 * ca1 * sa2 + a2 * sa3
    G2 = a2 * ca1 * sa3 + a3 * sa2
    H2 = -R3 * ca1 * sa2 * sa3
    L2 = R1 * (ca1 * ca2 + ca3) + R2 * (ca2 + ca1 * ca3) + R3 * (1 + ca1 * ca2 * ca3)
    
    # Coefficients for theta3 equation
    A3 = a1 * ca2 * sa3 + a3 * sa1
    B3 = -(R2 * sa1 * ca2 * sa3 + R3 * sa1 * sa3)
    C3 = a2 * ca1 * sa3 + a3 * sa2
    D3 = -(R2 * ca1 * sa2 * sa3 + R1 * sa2 * ca3)
    E3 = R2 * sa1 * sa2
    F3 = a1 * sa2 * ca3 + a2 * sa1
    G3 = a2 * sa1 * ca3 + a1 * sa2
    H3 = -R2 * sa1 * sa2 * ca3
    L3 = R1 * (ca1 * ca2 + ca3) + R2 * (1 + ca1 * ca2 * ca3) + R3 * (ca2 + ca1 * ca3)
    
    coeff2 = {'A': A2, 'B': B2, 'C': C2, 'D': D2, 'E': E2, 'F': F2, 'G': G2, 'H': H2, 'L': L2}
    coeff3 = {'A': A3, 'B': B3, 'C': C3, 'D': D3, 'E': E3, 'F': F3, 'G': G3, 'H': H3, 'L': L3}
    
    return coeff2, coeff3


def compute_PQR(theta1, coeff):
    """
    Compute P, Q, R for the standard trigonometric form.
    
    P*sin(theta) + Q*cos(theta) + R = 0
    """
    s1, c1 = np.sin(theta1), np.cos(theta1)
    
    P = coeff['C'] + coeff['E'] * s1 + coeff['G'] * c1
    Q = coeff['D'] + coeff['F'] * s1 + coeff['H'] * c1
    R = coeff['A'] * s1 + coeff['B'] * c1 + coeff['L']
    
    return P, Q, R


def solve_angle(theta1, coeff, sign=+1):
    """
    Solve P*sin(θ) + Q*cos(θ) + R = 0 using the half-angle substitution.
    
    With t = tan(θ/2):
        sin(θ) = 2t/(1+t²)
        cos(θ) = (1-t²)/(1+t²)
    
    This transforms to the quadratic:
        (R - Q)t² + 2Pt + (R + Q) = 0
    
    Solution:
        t = (-P ± sqrt(P² + Q² - R²)) / (R - Q)
        θ = 2 * arctan(t)
    
    Parameters:
    -----------
    theta1 : float or array
        Input angle in radians
    coeff : dict
        Coefficients dictionary
    sign : int
        +1 for positive root, -1 for negative root
    
    Returns:
    --------
    theta : float or array
        Output angle in radians
    valid : bool or array
        Whether the solution is valid
    """
    P, Q, R = compute_PQR(theta1, coeff)
    
    # Discriminant (must be >= 0 for real solutions)
    disc = P**2 + Q**2 - R**2
    
    valid = disc >= 0
    disc_safe = np.maximum(disc, 0)
    
    # Compute t = tan(θ/2)
    numerator = -P + sign * np.sqrt(disc_safe)
    denominator = R - Q
    
    # Handle denominator ≈ 0
    t = np.where(np.abs(denominator) > 1e-10, 
                 numerator / denominator,
                 np.where(np.abs(P) > 1e-10, -R/P, 0))
    
    # θ = 2 * arctan(t)
    theta = 2 * np.arctan(t)
    
    return theta, valid


def solve_bricard(theta1, a1, a2, a3, alpha1, alpha2, alpha3, R1, R2, R3):
    """
    Solve the Bricard 6R linkage for given input angle theta1.
    
    Returns theta2 and theta3 for both Form I and Form II.
    """
    # Convert alphas to radians
    alpha1_rad = np.radians(alpha1)
    alpha2_rad = np.radians(alpha2)
    alpha3_rad = np.radians(alpha3)
    
    # Compute coefficients
    coeff2, coeff3 = compute_coefficients(a1, a2, a3, alpha1_rad, alpha2_rad, alpha3_rad, R1, R2, R3)
    
    # Convert theta1 to radians
    theta1_rad = np.radians(theta1)
    
    # Form I: positive root for theta2, negative root for theta3
    theta2_I, valid2_I = solve_angle(theta1_rad, coeff2, sign=+1)
    theta3_I, valid3_I = solve_angle(theta1_rad, coeff3, sign=-1)
    
    # Form II: negative root for theta2, positive root for theta3
    theta2_II, valid2_II = solve_angle(theta1_rad, coeff2, sign=-1)
    theta3_II, valid3_II = solve_angle(theta1_rad, coeff3, sign=+1)
    
    # Convert back to degrees
    results = {
        'Form I': {
            'theta2': np.degrees(theta2_I),
            'theta3': np.degrees(theta3_I),
            'valid': valid2_I & valid3_I
        },
        'Form II': {
            'theta2': np.degrees(theta2_II),
            'theta3': np.degrees(theta3_II),
            'valid': valid2_II & valid3_II
        }
    }
    
    return results


def compute_kinematic_curves(a1, a2, a3, alpha1, alpha2, alpha3, R1, R2, R3, n_points=361):
    """
    Compute the kinematic curves (theta2 and theta3 vs theta1) for both forms.
    """
    theta1_range = np.linspace(-180, 180, n_points)
    
    results = solve_bricard(theta1_range, a1, a2, a3, alpha1, alpha2, alpha3, R1, R2, R3)
    
    return theta1_range, results


# ============================================================================
# Streamlit Interface
# ============================================================================

# Sidebar for parameters
st.sidebar.header("🔧 Link Parameters")

st.sidebar.subheader("Link 1 & 4")
col1, col2, col3 = st.sidebar.columns(3)
a1 = col1.number_input("a₁", value=2.4, step=0.1, format="%.2f")
alpha1 = col2.number_input("α₁ (°)", value=40.0, step=5.0, format="%.1f")
R1 = col3.number_input("R₁", value=0.5, step=0.1, format="%.2f")

st.sidebar.subheader("Link 2 & 5")
col1, col2, col3 = st.sidebar.columns(3)
a2 = col1.number_input("a₂", value=2.9, step=0.1, format="%.2f")
alpha2 = col2.number_input("α₂ (°)", value=80.0, step=5.0, format="%.1f")
R2 = col3.number_input("R₂", value=0.55, step=0.1, format="%.2f")

st.sidebar.subheader("Link 3 & 6")
col1, col2, col3 = st.sidebar.columns(3)
a3 = col1.number_input("a₃", value=1.5, step=0.1, format="%.2f")
alpha3 = col2.number_input("α₃ (°)", value=110.0, step=5.0, format="%.1f")
R3 = col3.number_input("R₃", value=0.42, step=0.1, format="%.2f")

# Main content
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("🎛️ Input Angle")
    theta1 = st.slider("θ₁ (degrees)", min_value=-180.0, max_value=180.0, value=0.0, step=1.0)
    
    # Solve for current theta1
    results = solve_bricard(theta1, a1, a2, a3, alpha1, alpha2, alpha3, R1, R2, R3)
    
    # Display Form I results
    st.markdown("---")
    st.subheader("📐 Closure Form I")
    form1 = results['Form I']
    if form1['valid']:
        st.success("✓ VALID")
        col1, col2 = st.columns(2)
        col1.metric("θ₂", f"{form1['theta2']:.1f}°")
        col2.metric("θ₃", f"{form1['theta3']:.1f}°")
    else:
        st.error("✗ INVALID")
        st.write("No real solution exists for this θ₁")
    
    # Display Form II results
    st.markdown("---")
    st.subheader("📐 Closure Form II")
    form2 = results['Form II']
    if form2['valid']:
        st.success("✓ VALID")
        col1, col2 = st.columns(2)
        col1.metric("θ₂", f"{form2['theta2']:.1f}°")
        col2.metric("θ₃", f"{form2['theta3']:.1f}°")
    else:
        st.error("✗ INVALID")
        st.write("No real solution exists for this θ₁")

with col_right:
    st.subheader("📈 Kinematic Curves")
    
    # Compute full kinematic curves
    theta1_range, curves = compute_kinematic_curves(a1, a2, a3, alpha1, alpha2, alpha3, R1, R2, R3)
    
    # Create subplot figure
    fig = make_subplots(rows=1, cols=1)
    
    # Form I - theta2 (solid blue)
    theta2_I = curves['Form I']['theta2']
    valid_I = curves['Form I']['valid']
    theta2_I_plot = np.where(valid_I, theta2_I, np.nan)
    fig.add_trace(go.Scatter(
        x=theta1_range, y=theta2_I_plot,
        mode='lines', name='Form I (θ₂)',
        line=dict(color='blue', width=2)
    ))
    
    # Form I - theta3 (dashed blue)
    theta3_I = curves['Form I']['theta3']
    theta3_I_plot = np.where(valid_I, theta3_I, np.nan)
    fig.add_trace(go.Scatter(
        x=theta1_range, y=theta3_I_plot,
        mode='lines', name='Form I (θ₃)',
        line=dict(color='blue', width=2, dash='dash')
    ))
    
    # Form II - theta2 (solid purple)
    theta2_II = curves['Form II']['theta2']
    valid_II = curves['Form II']['valid']
    theta2_II_plot = np.where(valid_II, theta2_II, np.nan)
    fig.add_trace(go.Scatter(
        x=theta1_range, y=theta2_II_plot,
        mode='lines', name='Form II (θ₂)',
        line=dict(color='purple', width=2)
    ))
    
    # Form II - theta3 (dashed purple)
    theta3_II = curves['Form II']['theta3']
    theta3_II_plot = np.where(valid_II, theta3_II, np.nan)
    fig.add_trace(go.Scatter(
        x=theta1_range, y=theta3_II_plot,
        mode='lines', name='Form II (θ₃)',
        line=dict(color='purple', width=2, dash='dash')
    ))
    
    # Add current position markers
    if form1['valid']:
        fig.add_trace(go.Scatter(
            x=[theta1], y=[form1['theta2']],
            mode='markers', name='Current (Form I)',
            marker=dict(color='blue', size=12, symbol='circle'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[theta1], y=[form1['theta3']],
            mode='markers',
            marker=dict(color='blue', size=12, symbol='diamond'),
            showlegend=False
        ))
    
    if form2['valid']:
        fig.add_trace(go.Scatter(
            x=[theta1], y=[form2['theta2']],
            mode='markers', name='Current (Form II)',
            marker=dict(color='purple', size=12, symbol='circle'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[theta1], y=[form2['theta3']],
            mode='markers',
            marker=dict(color='purple', size=12, symbol='diamond'),
            showlegend=False
        ))
    
    # Add vertical line at current theta1
    fig.add_vline(x=theta1, line_dash="dot", line_color="gray", opacity=0.5)
    
    # Layout
    fig.update_layout(
        xaxis_title="Input Angle θ₁ (degrees)",
        yaxis_title="Output Angles θ₂, θ₃ (degrees)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=500,
        hovermode='x unified'
    )
    
    fig.update_xaxes(range=[-180, 180], dtick=45)
    fig.update_yaxes(range=[-180, 180], dtick=45)
    
    st.plotly_chart(fig, use_container_width=True)

# Additional info
st.markdown("---")
st.subheader("📋 Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Symmetry Relations:**")
    st.latex(r"\theta_4 = \theta_1, \quad \theta_5 = \theta_2, \quad \theta_6 = \theta_3")

with col2:
    st.markdown("**Current Configuration:**")
    st.write(f"θ₁ = θ₄ = {theta1:.1f}°")
    if form1['valid']:
        st.write(f"Form I: θ₂ = θ₅ = {form1['theta2']:.1f}°, θ₃ = θ₆ = {form1['theta3']:.1f}°")
    if form2['valid']:
        st.write(f"Form II: θ₂ = θ₅ = {form2['theta2']:.1f}°, θ₃ = θ₆ = {form2['theta3']:.1f}°")

with col3:
    st.markdown("**Solution Method:**")
    st.latex(r"\theta = \psi \pm \arccos\left(\frac{-R}{\sqrt{P^2+Q^2}}\right)")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>A Tensor Method for the Kinematical Analysis of the Line-Symmetric Bricard 6R Linkage.</p>
    <p>Author: Ayman Alashkar</p>
    <p>Mechanics and Materials Unit, OIST | December 2025</p>
</div>
""", unsafe_allow_html=True)
