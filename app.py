import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioPredict | Heart Risk Dashboard",
    page_icon="🫀",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: #f0f4f8;
}

/* Top header bar */
.top-bar {
    background: linear-gradient(135deg, #1a56db, #1e40af);
    border-radius: 16px;
    padding: 1.4rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(26,86,219,0.3);
}
.top-bar h1 { color: white; margin: 0; font-size: 1.6rem; font-weight: 700; }
.top-bar p  { color: rgba(255,255,255,0.75); margin: 2px 0 0; font-size: 0.85rem; }
.top-badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    color: white;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
}

/* Section cards */
.section-card {
    background: white;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #e2e8f0;
}
.section-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #1a56db;
    margin-bottom: 1rem;
}

/* Stat boxes */
.stat-row { display: flex; gap: 10px; margin-top: 10px; }
.stat-box {
    flex: 1;
    background: #f8faff;
    border: 1px solid #dbeafe;
    border-radius: 10px;
    padding: 10px 12px;
    text-align: center;
}
.stat-val { font-size: 1.3rem; font-weight: 700; color: #1e3a8a; }
.stat-lbl { font-size: 0.68rem; color: #64748b; margin-top: 2px; }

/* Result cards */
.result-high {
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border: 1.5px solid #fca5a5;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
}
.result-low {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 1.5px solid #86efac;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
}
.result-title-high { color: #dc2626; font-size: 1.3rem; font-weight: 700; margin: 0 0 6px; }
.result-title-low  { color: #16a34a; font-size: 1.3rem; font-weight: 700; margin: 0 0 6px; }
.result-sub { color: #475569; font-size: 0.85rem; margin: 0; }

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #1a56db, #1e40af) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.7rem 1rem !important;
    width: 100% !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 14px rgba(26,86,219,0.35) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e40af, #1e3a8a) !important;
    transform: translateY(-1px) !important;
}

/* Input styling */
.stNumberInput input, .stSelectbox div {
    border-radius: 8px !important;
    border-color: #cbd5e1 !important;
    background: #f8faff !important;
}

label { color: #374151 !important; font-size: 0.83rem !important; font-weight: 500 !important; }

/* Disclaimer */
.disclaimer {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.78rem;
    color: #92400e;
    margin-top: 1rem;
    text-align: center;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 0.7rem;
    color: #94a3b8;
    margin-top: 2rem;
    padding-bottom: 1rem;
}

/* Hide streamlit default header */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('cardio_rf_model.pkl')

model = load_model()

# ── Top Bar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
    <div>
        <h1>🫀 CardioPredict</h1>
        <p>AI-powered cardiovascular disease risk assessment dashboard</p>
    </div>
    <div class="top-badge">Random Forest · 96.8% Accuracy</div>
</div>
""", unsafe_allow_html=True)

# ── Layout: Left inputs | Right dashboard ────────────────────────────────────
left, right = st.columns([1.1, 1], gap="large")

with left:
    # Personal Info
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age    = st.number_input("Age (years)", min_value=10, max_value=100, value=40)
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=165)
    with c2:
        gender = st.selectbox("Gender", [1, 2],
                     format_func=lambda x: "Female 👩" if x==1 else "Male 👨")
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0)

    bmi = round(weight / ((height/100)**2), 1)
    bmi_cat = ("Underweight" if bmi < 18.5
          else "Normal ✅" if bmi < 25
          else "Overweight ⚠️" if bmi < 30
          else "Obese 🔴")

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-box"><div class="stat-val">{bmi}</div><div class="stat-lbl">BMI</div></div>
        <div class="stat-box"><div class="stat-val" style="font-size:0.95rem">{bmi_cat}</div><div class="stat-lbl">Category</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Vitals
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">❤️ Blood Pressure & Vitals</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        ap_hi = st.number_input("Systolic BP — ap_hi", min_value=60, max_value=250, value=120)
        cholesterol = st.selectbox("Cholesterol", [1,2,3],
                         format_func=lambda x: {1:"Normal ✅",2:"Above Normal ⚠️",3:"Well Above 🔴"}[x])
    with c4:
        ap_lo = st.number_input("Diastolic BP — ap_lo", min_value=40, max_value=200, value=80)
        gluc  = st.selectbox("Glucose", [1,2,3],
                    format_func=lambda x: {1:"Normal ✅",2:"Above Normal ⚠️",3:"Well Above 🔴"}[x])

    bp_status = ("Normal ✅" if ap_hi < 120
            else "Elevated ⚠️" if ap_hi < 130
            else "Stage 1 ⚠️" if ap_hi < 140
            else "Stage 2 🔴")
    pulse_p = ap_hi - ap_lo

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-box"><div class="stat-val">{ap_hi}/{ap_lo}</div><div class="stat-lbl">BP (mmHg)</div></div>
        <div class="stat-box"><div class="stat-val" style="font-size:0.9rem">{bp_status}</div><div class="stat-lbl">BP Status</div></div>
        <div class="stat-box"><div class="stat-val">{pulse_p}</div><div class="stat-lbl">Pulse Pressure</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Lifestyle
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏃 Lifestyle Factors</div>', unsafe_allow_html=True)
    c5, c6, c7 = st.columns(3)
    with c5:
        smoke  = st.selectbox("Smoking 🚬", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    with c6:
        alco   = st.selectbox("Alcohol 🍺", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    with c7:
        active = st.selectbox("Active 💪",  [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    st.markdown('</div>', unsafe_allow_html=True)

    # Predict button
    predict_clicked = st.button("🔍 Analyse My Cardiovascular Risk", use_container_width=True)

with right:
    # ── Default Dashboard Charts ──────────────────────────────────────────────
    bmi_val = round(weight / ((height/100)**2), 1)

    # Gauge chart — BMI
    fig_bmi = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bmi_val,
        title={'text': "BMI", 'font': {'size': 14, 'color': '#1e3a8a'}},
        number={'font': {'size': 28, 'color': '#1e3a8a'}},
        gauge={
            'axis': {'range': [10, 45], 'tickwidth': 1, 'tickcolor': '#94a3b8'},
            'bar': {'color': '#1a56db'},
            'bgcolor': '#f0f4f8',
            'steps': [
                {'range': [10, 18.5], 'color': '#bfdbfe'},
                {'range': [18.5, 25], 'color': '#bbf7d0'},
                {'range': [25, 30],   'color': '#fef08a'},
                {'range': [30, 45],   'color': '#fecaca'},
            ],
            'threshold': {'line': {'color': '#1e3a8a', 'width': 3}, 'thickness': 0.75, 'value': bmi_val}
        }
    ))
    fig_bmi.update_layout(
        height=200, margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family='Inter')
    )

    # Gauge chart — Systolic BP
    fig_bp = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ap_hi,
        title={'text': "Systolic BP (mmHg)", 'font': {'size': 14, 'color': '#1e3a8a'}},
        number={'font': {'size': 28, 'color': '#1e3a8a'}},
        gauge={
            'axis': {'range': [60, 200], 'tickwidth': 1, 'tickcolor': '#94a3b8'},
            'bar': {'color': '#1a56db'},
            'bgcolor': '#f0f4f8',
            'steps': [
                {'range': [60, 120],  'color': '#bbf7d0'},
                {'range': [120, 130], 'color': '#fef08a'},
                {'range': [130, 140], 'color': '#fed7aa'},
                {'range': [140, 200], 'color': '#fecaca'},
            ],
            'threshold': {'line': {'color': '#dc2626', 'width': 3}, 'thickness': 0.75, 'value': ap_hi}
        }
    ))
    fig_bp.update_layout(
        height=200, margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family='Inter')
    )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Health Metrics Dashboard</div>', unsafe_allow_html=True)
    gc1, gc2 = st.columns(2)
    with gc1:
        st.plotly_chart(fig_bmi, use_container_width=True, config={'displayModeBar': False})
    with gc2:
        st.plotly_chart(fig_bp,  use_container_width=True, config={'displayModeBar': False})

    # Risk factor bar chart
    factors = ['Age', 'BMI', 'Systolic BP', 'Cholesterol', 'Glucose']
    age_score  = min(age/100*100, 100)
    bmi_score  = min((bmi/40)*100, 100)
    bp_score   = min(((ap_hi-60)/140)*100, 100)
    chol_score = {1:20, 2:60, 3:90}[cholesterol]
    gluc_score = {1:20, 2:60, 3:90}[gluc]
    values = [age_score, bmi_score, bp_score, chol_score, gluc_score]
    colors = ['#ef4444' if v >= 70 else '#f59e0b' if v >= 40 else '#22c55e' for v in values]

    fig_bar = go.Figure(go.Bar(
        x=factors, y=values,
        marker_color=colors,
        text=[f"{v:.0f}" for v in values],
        textposition='outside',
        textfont=dict(size=11, color='#1e3a8a')
    ))
    fig_bar.update_layout(
        title=dict(text="Risk Factor Scores", font=dict(size=13, color='#1e3a8a')),
        yaxis=dict(range=[0, 110], showgrid=True, gridcolor='#f1f5f9',
                   title='Risk Score', titlefont=dict(size=11)),
        xaxis=dict(tickfont=dict(size=11)),
        height=240, margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family='Inter'),
        showlegend=False
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Prediction Result ─────────────────────────────────────────────────────
    if predict_clicked:
        input_df = pd.DataFrame([[
            age, gender, height, weight,
            ap_hi, ap_lo, cholesterol, gluc,
            smoke, alco, active, bmi_val
        ]], columns=[
            'age','gender','height','weight',
            'ap_hi','ap_lo','cholesterol','gluc',
            'smoke','alco','active','bmi'
        ])

        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        confidence  = round(max(probability)*100, 2)
        risk_score  = round(probability[1]*100, 1)

        # Donut chart — risk probability
        fig_donut = go.Figure(go.Pie(
            values=[risk_score, 100-risk_score],
            labels=['Risk', 'Safe'],
            hole=0.65,
            marker_colors=['#ef4444','#22c55e'] if prediction==1 else ['#22c55e','#e2e8f0'],
            textinfo='none',
            hoverinfo='label+percent'
        ))
        fig_donut.add_annotation(
            text=f"{risk_score}%",
            x=0.5, y=0.5, font_size=24,
            font_color='#ef4444' if prediction==1 else '#16a34a',
            showarrow=False, font=dict(family='Inter', weight=700)
        )
        fig_donut.add_annotation(
            text="Risk Score",
            x=0.5, y=0.35, font_size=11,
            font_color='#64748b', showarrow=False,
            font=dict(family='Inter')
        )
        fig_donut.update_layout(
            height=200, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor='white',
            showlegend=False,
            font=dict(family='Inter')
        )

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔍 Prediction Result</div>', unsafe_allow_html=True)

        rc1, rc2 = st.columns([1, 1])
        with rc1:
            st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})
        with rc2:
            if prediction == 1:
                st.markdown(f"""
                <div style="padding-top:1.5rem">
                <p class="result-title-high">⚠️ High Risk</p>
                <p class="result-sub">Our model detected a high likelihood of cardiovascular disease.</p>
                <div class="stat-row" style="margin-top:12px">
                    <div class="stat-box"><div class="stat-val" style="color:#dc2626">{confidence}%</div><div class="stat-lbl">Confidence</div></div>
                    <div class="stat-box"><div class="stat-val" style="color:#dc2626">{risk_score}%</div><div class="stat-lbl">Risk Score</div></div>
                </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="padding-top:1.5rem">
                <p class="result-title-low">✅ Low Risk</p>
                <p class="result-sub">Our model detected a low likelihood of cardiovascular disease.</p>
                <div class="stat-row" style="margin-top:12px">
                    <div class="stat-box"><div class="stat-val" style="color:#16a34a">{confidence}%</div><div class="stat-lbl">Confidence</div></div>
                    <div class="stat-box"><div class="stat-val" style="color:#16a34a">{risk_score}%</div><div class="stat-lbl">Risk Score</div></div>
                </div>
                </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Key findings
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔎 Key Findings</div>', unsafe_allow_html=True)
        if prediction == 1:
            if ap_hi >= 140: st.error(f"🔴 Systolic BP {ap_hi} mmHg — Stage 2 Hypertension")
            if cholesterol >= 2: st.warning("🟡 Cholesterol is above normal")
            if bmi_val >= 25: st.warning(f"🟡 BMI {bmi_val} — {bmi_cat}")
            if smoke == 1: st.error("🔴 Smoking significantly increases heart risk")
            if age >= 50: st.warning(f"🟡 Age {age} is a cardiovascular risk factor")
        else:
            if ap_hi < 120: st.success("✅ Blood pressure is in normal range")
            if bmi_val < 25: st.success(f"✅ BMI {bmi_val} is healthy")
            if smoke == 0: st.success("✅ Non-smoker — great for heart health")
            if active == 1: st.success("✅ Physically active — keeps heart strong")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="disclaimer">⚠️ This tool is for educational purposes only and is NOT a medical diagnosis. Always consult a qualified doctor.</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    CardioPredict · SYBCA Group ML Project · Streamlit + scikit-learn + Plotly · Random Forest Classifier
</div>
""", unsafe_allow_html=True)
