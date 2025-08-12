import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys

# Import robust modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from utils.robust_afrofuturistic import (
        RobustAfroAnalyzer, create_simple_3d_plot, create_population_comparison,
        calculate_ubuntu_stats, generate_synthetic_data, apply_afrofuturistic_theme,
        export_data_csv, export_data_json, AFRO_COLORS
    )
except ImportError:
    # Fallback if import fails
    AFRO_COLORS = {'gold': '#fbbf24', 'purple': '#7c3aed', 'amber': '#f59e0b'}

# Set page configuration first
st.set_page_config(
    page_title="Academic Research Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Advanced CSS styling with modern UI/UX
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');
    
    /* Global Variables */
    :root {
        --bg-main: #f8fafc;
        --bg-alt: #ffffff;
        --bg-card: rgba(59, 130, 246, 0.05);
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --accent-primary: #3b82f6;
        --accent-secondary: #1d4ed8;
        --border-primary: #e2e8f0;
        --blue-primary: #3b82f6;
        --blue-secondary: #1e40af;
    }
    
    /* Main App Background */
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #cbd5e1 100%);
        color: var(--text-primary);
        min-height: 100vh;
    }
    
    /* Hide Streamlit UI Elements */
    .stApp > header {
        background-color: transparent;
    }
    
    .stApp > div:first-child {
        background-color: transparent;
    }
    
    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(99, 102, 241, 0.05));
        border: 2px solid var(--accent-primary);
        border-radius: 16px;
        padding: 3rem 2rem;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.15);
        backdrop-filter: blur(5px);
    }
    
    .hero-title {
        color: var(--accent-primary);
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-weight: 700;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        color: var(--text-secondary);
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        line-height: 1.6;
        font-weight: 400;
    }
    
    /* Tool Cards */
    .tool-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .tool-card {
        background: var(--bg-alt);
        border: 1px solid var(--border-primary);
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .tool-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(251, 191, 36, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .tool-card:hover::before {
        left: 100%;
    }
    
    .tool-card:hover {
        transform: translateY(-2px);
        border-color: var(--accent-primary);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.2);
    }
    
    .tool-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: block;
        text-align: center;
    }
    
    .tool-title {
        color: var(--accent-primary);
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .tool-description {
        color: var(--text-secondary);
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-size: 0.9rem;
        line-height: 1.5;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .tool-features {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .tool-features li {
        color: var(--text-primary);
        font-size: 0.8rem;
        margin: 0.3rem 0;
        padding-left: 1rem;
        position: relative;
    }
    
    .tool-features li::before {
        content: '•';
        position: absolute;
        left: 0;
        color: var(--accent-primary);
    }
    
    /* Feature Highlights */
    .feature-section {
        background: var(--bg-alt);
        border: 1px solid var(--border-primary);
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.05);
    }
    
    .feature-title {
        color: var(--accent-primary);
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-weight: 600;
        font-size: 1.4rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .feature-item {
        background: rgba(59, 130, 246, 0.03);
        border: 1px solid rgba(59, 130, 246, 0.1);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .feature-item h4 {
        color: var(--accent-primary);
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .feature-item p {
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    /* Statistics Bar */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .stat-item {
        background: var(--bg-alt);
        border: 1px solid var(--border-primary);
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
    }
    
    .stat-number {
        color: var(--accent-primary);
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-weight: 700;
        font-size: 1.8rem;
        display: block;
    }
    
    .stat-label {
        color: var(--text-secondary);
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-subtitle {
            font-size: 1.1rem;
        }
        
        .tool-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* Button Styling */
    .stButton > button {
        background: var(--accent-primary);
        border: 1px solid var(--accent-primary);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.2s ease;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    .stButton > button:hover {
        background: var(--accent-secondary);
        border-color: var(--accent-secondary);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">Academic Research Platform</h1>
    <p class="hero-subtitle">
        Comprehensive pharmaceutical research with advanced computational methods<br>
        Statistical Analysis • Molecular Modeling • Population Studies • Meta-Analysis
    </p>
</div>
""", unsafe_allow_html=True)

# Platform Statistics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-item">
        <span class="stat-number">15+</span>
        <div class="stat-label">Research Tools</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-item">
        <span class="stat-number">3D</span>
        <div class="stat-label">Molecular Visualization</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-item">
        <span class="stat-number">AI</span>
        <div class="stat-label">Statistical Analytics</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-item">
        <span class="stat-number">ML</span>
        <div class="stat-label">Machine Learning</div>
    </div>
    """, unsafe_allow_html=True)

# Main Tool Navigation
st.markdown("""
<div class="feature-section">
    <h2 class="feature-title">Research Tools</h2>
</div>
""", unsafe_allow_html=True)

# Tool Grid Layout
tool_cols = st.columns(3)

# Row 1: Core Research Tools
with tool_cols[0]:
    if st.button("📊 Statistical Analysis", key="stat_analysis", use_container_width=True):
        st.switch_page("pages/1_Statistical_Analysis.py")
    st.markdown("""
    <div class="tool-card">
        <div class="tool-icon">📊</div>
        <div class="tool-title">Statistical Analysis</div>
        <div class="tool-description">Advanced meta-analysis and statistical modeling</div>
        <ul class="tool-features">
            <li>Random effects modeling</li>
            <li>Population-specific analysis</li>
            <li>Effect size calculations</li>
            <li>Heterogeneity assessment</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with tool_cols[1]:
    if st.button("🧬 Molecular Modeling", key="molecular", use_container_width=True):
        st.switch_page("pages/5_Molecular_Docking.py")
    st.markdown("""
    <div class="tool-card">
        <div class="tool-icon">🧬</div>
        <div class="tool-title">Molecular Modeling</div>
        <div class="tool-description">3D protein-drug interaction visualization</div>
        <ul class="tool-features">
            <li>Interactive 3D structures</li>
            <li>Docking simulations</li>
            <li>ADMET predictions</li>
            <li>Binding affinity analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with tool_cols[2]:
    if st.button("💊 Pharmacology", key="pharmacology", use_container_width=True):
        st.switch_page("pages/7_Pharmacological_Topological_Maps.py")
    st.markdown("""
    <div class="tool-card">
        <div class="tool-icon">💊</div>
        <div class="tool-title">Pharmacology Analysis</div>
        <div class="tool-description">Population-specific drug response modeling</div>
        <ul class="tool-features">
            <li>CYP450 analysis</li>
            <li>PK-PD modeling</li>
            <li>Therapeutic windows</li>
            <li>Genetic variations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Row 2: Data Processing Tools
tool_cols2 = st.columns(3)

with tool_cols2[0]:
    if st.button("📄 Data Import", key="data_import", use_container_width=True):
        st.switch_page("pages/4_Data_Import.py")
    st.markdown("""
    <div class="tool-card">
        <div class="tool-icon">📄</div>
        <div class="tool-title">Data Import</div>
        <div class="tool-description">Intelligent document processing and validation</div>
        <ul class="tool-features">
            <li>PDF text extraction</li>
            <li>Automatic validation</li>
            <li>Citation analysis</li>
            <li>Quality assessment</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with tool_cols2[1]:
    if st.button("✍️ Academic Writing", key="writing", use_container_width=True):
        st.switch_page("pages/3_Paper_Rewriter.py")
    st.markdown("""
    <div class="tool-card">
        <div class="tool-icon">✍️</div>
        <div class="tool-title">Academic Writing</div>
        <div class="tool-description">AI-enhanced research paper generation</div>
        <ul class="tool-features">
            <li>Systematic methodology</li>
            <li>Citation formatting</li>
            <li>Readability analysis</li>
            <li>Quality enhancement</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with tool_cols2[2]:
    if st.button("📊 Validation Results", key="validation", use_container_width=True):
        st.switch_page("pages/6_Validation_Results.py")
    st.markdown("""
    <div class="tool-card">
        <div class="tool-icon">📊</div>
        <div class="tool-title">Validation Results</div>
        <div class="tool-description">Comprehensive research validation dashboard</div>
        <ul class="tool-features">
            <li>Automated testing</li>
            <li>Statistical validation</li>
            <li>Reproducibility checks</li>
            <li>Quality metrics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Row 3: Advanced Tools
tool_cols3 = st.columns(3)

with tool_cols3[0]:
    if st.button("🌍 Population Analysis", key="population", use_container_width=True):
        st.switch_page("pages/2_Meta_Analysis_Verification.py")
    st.markdown("""
    <div class="tool-card">
        <div class="tool-icon">🌍</div>
        <div class="tool-title">Population Analysis</div>
        <div class="tool-description">Genetic diversity and drug response modeling</div>
        <ul class="tool-features">
            <li>Ethnic stratification</li>
            <li>Genetic polymorphisms</li>
            <li>Population pharmacokinetics</li>
            <li>Demographic analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with tool_cols3[1]:
    if st.button("⚡ Advanced Analytics", key="quantum", use_container_width=True):
        st.switch_page("pages/8_Data_Citations.py")
    st.markdown("""
    <div class="tool-card">
        <div class="tool-icon">⚡</div>
        <div class="tool-title">Advanced Analytics</div>
        <div class="tool-description">Enhanced computational methods and algorithms</div>
        <ul class="tool-features">
            <li>Trend analysis</li>
            <li>Advanced algorithms</li>
            <li>Enhanced precision</li>
            <li>Predictive modeling</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with tool_cols3[2]:
    if st.button("🎯 Research Methods", key="ubuntu", use_container_width=True):
        st.switch_page("pages/9_Plot_Implications.py")
    st.markdown("""
    <div class="tool-card">
        <div class="tool-icon">🎯</div>
        <div class="tool-title">Research Methods</div>
        <div class="tool-description">Systematic research methodologies and frameworks</div>
        <ul class="tool-features">
            <li>Holistic analysis</li>
            <li>Evidence-based methods</li>
            <li>Ethical frameworks</li>
            <li>Quality assurance</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Jupyter Integration
st.markdown("""
<div class="feature-section">
    <h2 class="feature-title">Computational Environments</h2>
</div>
""", unsafe_allow_html=True)

jupyter_cols = st.columns(2)

with jupyter_cols[0]:
    if st.button("🚀 JupyterLab Enhanced", key="jupyter_enhanced", use_container_width=True):
        st.switch_page("pages/10_Jupyter_Integration.py")
    st.markdown("""
    <div class="tool-card">
        <div class="tool-icon">🚀</div>
        <div class="tool-title">JupyterLab Enhanced</div>
        <div class="tool-description">Advanced computational environment with enhanced features</div>
        <ul class="tool-features">
            <li>Port 8889 - Enhanced widgets</li>
            <li>Molecular modeling tools</li>
            <li>Statistical analysis modules</li>
            <li>Advanced visualizations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with jupyter_cols[1]:
    st.markdown("""
    <div class="tool-card">
        <div class="tool-icon">📓</div>
        <div class="tool-title">Standard Jupyter</div>
        <div class="tool-description">Traditional notebook environment for compatibility</div>
        <ul class="tool-features">
            <li>Port 8888 - Classic interface</li>
            <li>Educational tutorials</li>
            <li>Quick prototyping</li>
            <li>Legacy notebook support</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 1rem;">
        <a href="http://localhost:8888/lab" target="_blank" style="text-decoration: none;">
            <button style="background: linear-gradient(45deg, #7c3aed, #a855f7); border: 2px solid #fbbf24; color: white; font-weight: bold; border-radius: 10px; padding: 0.5rem 1rem; cursor: pointer; font-family: 'Exo 2', sans-serif;">
                🔗 Open Standard Jupyter
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

# Platform Features
st.markdown("""
<div class="feature-section">
    <h2 class="feature-title">Platform Capabilities</h2>
    <div class="feature-grid">
        <div class="feature-item">
            <h4>Research Excellence</h4>
            <p>Comprehensive tools for systematic reviews, meta-analysis, and molecular research with rigorous statistical methods</p>
        </div>
        <div class="feature-item">
            <h4>Global Accessibility</h4>
            <p>Cloud deployment ready with Streamlit Cloud, Docker, and Kubernetes support for worldwide research collaboration</p>
        </div>
        <div class="feature-item">
            <h4>Advanced Analytics</h4>
            <p>Enhanced computational methods combining machine learning with cutting-edge statistical algorithms</p>
        </div>
        <div class="feature-item">
            <h4>Molecular Precision</h4>
            <p>3D visualization, protein docking, and population-specific drug response modeling for personalized medicine</p>
        </div>
        <div class="feature-item">
            <h4>Statistical Rigor</h4>
            <p>Advanced meta-analysis, effect size calculations, and heterogeneity assessment with demographic considerations</p>
        </div>
        <div class="feature-item">
            <h4>Quality Assurance</h4>
            <p>Systematic methodology framework ensuring research maintains highest academic standards and reproducibility</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="feature-section" style="text-align: center; margin-top: 3rem;">
    <h3 style="color: var(--accent-primary); font-family: 'Inter', 'Segoe UI', sans-serif; margin-bottom: 1rem;">
        Academic Research Platform
    </h3>
    <p style="color: var(--text-secondary); font-family: 'Inter', 'Segoe UI', sans-serif; line-height: 1.6;">
        Built by David Joshua Ferguson, BS, MS, PharmD Candidate, RSci MRSB MRSC<br>
        Advanced computational tools for pharmaceutical research and meta-analysis<br>
        <strong>Evidence-based research for global health improvement</strong>
    </p>
</div>
""", unsafe_allow_html=True)