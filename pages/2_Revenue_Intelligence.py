"""
pages/2_Revenue_Intelligence.py — ALO Platform · Revenue & WTP Intelligence
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_embedded import load_raw
from engine import prepare_data, run_clustering, run_classification, COLORS, PLOTLY_LAYOUT, GLOBAL_CSS, PERSONA_PALETTE

st.set_page_config(page_title="Revenue Intelligence · ALO", page_icon="◈", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""<div style="font-size:1rem;font-weight:700;color:#E2E8F0;padding:16px 0 12px;border-bottom:1px solid #1E2D45;margin-bottom:16px;">◈ Revenue Intelligence</div>""", unsafe_allow_html=True)
    model_choice = st.radio("Prediction model", ["Random Forest", "Logistic Regression"])
    show_high_only = st.toggle("Highlight high-WTP users only", False)
    st.markdown("---")
    st.markdown("""<div style="font-size:0.72rem;color:#475569;">
    Random Forest: ensemble, captures non-linear patterns.<br>
    Logistic Regression: linear, more interpretable, higher accuracy on this dataset.
    </div>""", unsafe_allow_html=True)

@st.cache_data
def get_revenue_data():
    raw = load_raw()
    df  = prepare_data(raw)
    df_c, _ = run_clustering(df, k=4)
    return run_classification(df_c), df_c

res, df_c = get_revenue_data()

# ── Header ────────────────────────────────────────────────────────────────────
wtp_med = res['wtp_med']
rf, lr  = res['rf'], res['lr']

st.markdown(f"""
<div style="padding:0 0 24px 0; border-bottom:1px solid #1E2D45; margin-bottom:24px;">
    <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; color:#F59E0B; font-weight:600; margin-bottom:6px;">Revenue Intelligence</div>
    <h1 style="margin:0; font-size:1.8rem; font-weight:700; color:#E2E8F0; letter-spacing:-0.02em;">Who Will Pay?</h1>
    <p style="margin:6px 0 0 0; color:#64748B; font-size:0.88rem;">WTP classification · Predicting subscribers above ${wtp_med:.0f}/mo · n=450 users</p>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
active = rf if model_choice == "Random Forest" else lr
c1.metric("Model Accuracy",    f"{active['acc']*100:.1f}%",    f"{model_choice}")
c2.metric("AUC-ROC",          f"{active['auc']:.3f}",          "vs 0.500 baseline")
c3.metric("WTP Threshold",     f"${wtp_med:.0f}/mo",           "Median split")
c4.metric("High-WTP Users",    f"{res['df']['WTP_High'].sum()}", f"{res['df']['WTP_High'].mean()*100:.0f}% of cohort")
c5.metric("RF vs LR Gap",      f"{abs(rf['acc']-lr['acc'])*100:.1f}pp", "LR leads on accuracy")

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Row 1: ROC + Confusion ────────────────────────────────────────────────────
col1, col2 = st.columns([5, 5])

with col1:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>ROC Curves · Model Discriminative Power</div>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rf['fpr'], y=rf['tpr'],
        mode='lines', name=f"Random Forest (AUC {rf['auc']:.3f})",
        line=dict(color=COLORS['blue'], width=3),
        fill='tozeroy', fillcolor='rgba(59,130,246,0.06)',
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra>Random Forest</extra>'
    ))
    fig.add_trace(go.Scatter(
        x=lr['fpr'], y=lr['tpr'],
        mode='lines', name=f"Logistic Regression (AUC {lr['auc']:.3f})",
        line=dict(color=COLORS['green'], width=3, dash='dash'),
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra>Logistic Reg</extra>'
    ))
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode='lines',
        name='Random (AUC 0.500)',
        line=dict(color=COLORS['muted'], width=1.5, dash='dot'),
    ))
    # Highlight "operating point" at FPR~0.3
    for model_data, color, name in [(rf, COLORS['blue'], 'RF'), (lr, COLORS['green'], 'LR')]:
        idx = np.argmin(np.abs(np.array(model_data['fpr']) - 0.3))
        fig.add_trace(go.Scatter(
            x=[model_data['fpr'][idx]], y=[model_data['tpr'][idx]],
            mode='markers', marker=dict(size=12, color=color,
                                         symbol='circle', line=dict(width=2,color='white')),
            showlegend=False, hovertemplate=f'{name} @ FPR=30%: TPR={model_data["tpr"][idx]:.2f}<extra></extra>'
        ))

    fig.update_layout(**PLOTLY_LAYOUT, title='ROC Curves — WTP Prediction',
                       height=380, xaxis_title='False Positive Rate (1 - Specificity)',
                       yaxis_title='True Positive Rate (Sensitivity)',
                       legend=dict(x=0.4, y=0.12, font=dict(size=11)))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown(f"""<div class="insight-card">
    <div class="label">◈ Model Insight</div>
    At a <b>30% false positive rate</b> (realistic campaign targeting), the models capture
    <b>{lr['tpr'][np.argmin(np.abs(np.array(lr['fpr'])-0.3))]*100:.0f}% of all high-WTP users</b>.
    For a growth campaign targeting 200 users, this means ~{int(200*lr['tpr'][np.argmin(np.abs(np.array(lr['fpr'])-0.3))])}
    genuine conversions — a strong signal for CAC-efficient growth.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>Confusion Matrix · {model_choice}</div>", unsafe_allow_html=True)

    cm = active['cm']
    labels = ['Low WTP','High WTP']

    # Annotate with business framing
    annot = [
        [f"<b>{cm[0,0]}</b><br><span style='font-size:9px;color:#94A3B8'>True Negatives<br>Correctly excluded</span>",
         f"<b>{cm[0,1]}</b><br><span style='font-size:9px;color:#94A3B8'>False Positives<br>Over-targeted</span>"],
        [f"<b>{cm[1,0]}</b><br><span style='font-size:9px;color:#94A3B8'>False Negatives<br>Missed revenue</span>",
         f"<b>{cm[1,1]}</b><br><span style='font-size:9px;color:#94A3B8'>True Positives<br>Revenue captured</span>"],
    ]
    cell_colors = [
        [COLORS['card2'], '#451A03'],
        ['#0A2540', '#064E3B'],
    ]

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels, y=labels,
        colorscale=[[0,'#0F2044'],[0.5,'#1D4ED8'],[1,'#3B82F6']],
        showscale=False,
        hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))
    # Add text annotations
    for i in range(2):
        for j in range(2):
            fig.add_annotation(x=labels[j], y=labels[i],
                                text=f"<b>{cm[i,j]}</b>",
                                showarrow=False,
                                font=dict(size=24, family='JetBrains Mono', color='white'))

    fig.update_layout(**PLOTLY_LAYOUT, height=300,
                       title=f'{model_choice} — Accuracy {active["acc"]*100:.1f}%',
                       xaxis_title='Predicted', yaxis_title='Actual',
                       margin=dict(l=60,r=10,t=40,b=60))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Business framing
    tp, fn = cm[1,1], cm[1,0]
    fp, tn = cm[0,1], cm[0,0]
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px;">
        <div style="background:#064E3B;border-radius:8px;padding:10px 14px;text-align:center;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.4rem;color:#34D399;font-weight:700;">{tp}</div>
            <div style="font-size:0.72rem;color:#6EE7B7;">Revenue captured ✓</div>
        </div>
        <div style="background:#450A0A;border-radius:8px;padding:10px 14px;text-align:center;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.4rem;color:#F87171;font-weight:700;">{fn}</div>
            <div style="font-size:0.72rem;color:#FCA5A5;">Missed revenue ✗</div>
        </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Row 2: Feature importance + WTP by segment ───────────────────────────────
col1, col2 = st.columns([5, 5])

with col1:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>What Predicts Payment Intent · Feature Importance</div>", unsafe_allow_html=True)

    top_n = st.slider("Features to show", 6, 18, 10, label_visibility='collapsed')
    imp = res['importances'].head(top_n).reset_index()
    imp.columns = ['Feature','Importance']
    imp['Feature'] = imp['Feature'].str.replace('_',' ').str.replace('enc','')
    imp_sorted = imp.sort_values('Importance')

    # Color top 3 differently
    colors_imp = [COLORS['amber'] if i >= len(imp_sorted)-3 else COLORS['blue']
                  for i in range(len(imp_sorted))]

    fig = go.Figure(go.Bar(
        y=imp_sorted['Feature'],
        x=imp_sorted['Importance'],
        orientation='h',
        marker_color=colors_imp,
        marker_opacity=0.9,
        text=[f"{v:.3f}" for v in imp_sorted['Importance']],
        textposition='outside',
        textfont=dict(size=10, family='JetBrains Mono', color=COLORS['muted']),
        hovertemplate='%{y}<br>Importance: %{x:.4f}<extra></extra>'
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                       title=f'Random Forest Feature Importance (Top {top_n})',
                       height=max(320, top_n*32), margin=dict(l=0,r=70,t=40,b=0),
                       xaxis_title='Gini Importance')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    top3 = res['importances'].head(3).index.str.replace('_',' ').tolist()
    st.markdown(f"""<div class="insight-card">
    <div class="label">◈ Monetisation Signal</div>
    <b>{top3[0]}, {top3[1]}, and {top3[2]}</b> are the top payment predictors — all
    outcome variables, not demographics. <b>Students pay when they see results,
    not because of who they are.</b> This validates a results-first, paywall-second
    product strategy: show value early, then convert.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>WTP Distribution · Drill into Segments</div>", unsafe_allow_html=True)

    df_wtp = res['df']
    if show_high_only:
        df_wtp_plot = df_wtp[df_wtp['WTP_High']==1]
        title_suffix = " (High WTP only)"
    else:
        df_wtp_plot = df_wtp
        title_suffix = ""

    # Violin plot — much more informative than box
    personas_sorted = df_wtp_plot.groupby('Persona')['Willingness_to_Pay'].median().sort_values(ascending=False).index.tolist()

    fig = go.Figure()
    for persona in personas_sorted:
        sub = df_wtp_plot[df_wtp_plot['Persona']==persona]['Willingness_to_Pay']
        color = PERSONA_PALETTE.get(persona, '#888')
        r,g,b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
        fig.add_trace(go.Violin(
            y=sub, name=persona,
            box_visible=True,
            meanline_visible=True,
            fillcolor=f'rgba({r},{g},{b},0.15)',
            line_color=color,
            meanline=dict(color='white', width=2),
            points='outliers',
            marker=dict(color=color, size=3, opacity=0.5),
            hovertemplate=f'<b>{persona}</b><br>WTP: $%{{y:.0f}}/mo<extra></extra>'
        ))

    fig.add_hline(y=wtp_med, line_color=COLORS['amber'], line_dash='dash', line_width=2,
                   annotation_text=f"Median ${wtp_med:.0f}", annotation_font_color=COLORS['amber'])
    fig.update_layout(**PLOTLY_LAYOUT,
                       title=f'WTP by Segment{title_suffix}',
                       yaxis_title='Willingness to Pay ($/mo)',
                       height=380, margin=dict(l=0,r=10,t=40,b=0),
                       xaxis=dict(showgrid=False))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Pricing recommendation callout
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:4px;">
        <div style="background:#111827;border:1px solid #1E2D45;border-top:2px solid #EF4444;border-radius:8px;padding:10px;text-align:center;">
            <div style="font-size:0.65rem;color:#64748B;text-transform:uppercase;letter-spacing:0.08em;">Entry Tier</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;color:#F87171;font-weight:700;">$35/mo</div>
            <div style="font-size:0.7rem;color:#475569;">Developing Strivers</div>
        </div>
        <div style="background:#111827;border:1px solid #1E2D45;border-top:2px solid #3B82F6;border-radius:8px;padding:10px;text-align:center;">
            <div style="font-size:0.65rem;color:#64748B;text-transform:uppercase;letter-spacing:0.08em;">Core Tier</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;color:#60A5FA;font-weight:700;">$45/mo</div>
            <div style="font-size:0.7rem;color:#475569;">High-Stress Engagers</div>
        </div>
        <div style="background:#111827;border:1px solid #1E2D45;border-top:2px solid #10B981;border-radius:8px;padding:10px;text-align:center;">
            <div style="font-size:0.65rem;color:#64748B;text-transform:uppercase;letter-spacing:0.08em;">Premium Tier</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;color:#34D399;font-weight:700;">$55/mo</div>
            <div style="font-size:0.7rem;color:#475569;">Efficient Achievers</div>
        </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
