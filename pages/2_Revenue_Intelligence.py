"""pages/2_Revenue_Intelligence.py — ALO Platform · Revenue & WTP Intelligence"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_embedded import load_raw
from engine import prepare_data, run_clustering, run_classification, COLORS, GLOBAL_CSS, PERSONA_PALETTE, theme

st.set_page_config(page_title="Revenue Intelligence · ALO", page_icon="💰", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div style='font-size:1rem;font-weight:700;color:#E2E8F0;padding:16px 0 12px;border-bottom:1px solid #1E2D45;margin-bottom:16px;'>Revenue Intelligence</div>", unsafe_allow_html=True)
    model_choice = st.radio("Prediction model", ["Random Forest","Logistic Regression"])
    show_high_only = st.toggle("High-WTP users only", False)

@st.cache_data
def get_data():
    df  = prepare_data(load_raw())
    df_c,_ = run_clustering(df,k=4)
    return run_classification(df_c),df_c

res,df_c = get_data()
rf,lr    = res['rf'],res['lr']
active   = rf if model_choice=="Random Forest" else lr
wtp_med  = res['wtp_med']

st.markdown(f"""<div style="padding:0 0 24px 0;border-bottom:1px solid #1E2D45;margin-bottom:24px;">
    <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.12em;color:#F59E0B;font-weight:600;margin-bottom:6px;">Revenue Intelligence</div>
    <h1 style="margin:0;font-size:1.8rem;font-weight:700;color:#E2E8F0;">Who Will Pay?</h1>
    <p style="margin:6px 0 0 0;color:#64748B;font-size:0.88rem;">WTP classification · threshold ${wtp_med:.0f}/mo · n=450 users</p>
</div>""", unsafe_allow_html=True)

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Model Accuracy",   f"{active['acc']*100:.1f}%",   model_choice)
c2.metric("AUC-ROC",          f"{active['auc']:.3f}",        "vs 0.500 baseline")
c3.metric("WTP Threshold",    f"${wtp_med:.0f}/mo",          "median split")
c4.metric("High-WTP Users",   f"{res['df']['WTP_High'].sum()}", f"{res['df']['WTP_High'].mean()*100:.0f}% of cohort")
c5.metric("LR vs RF Gap",     f"{abs(rf['acc']-lr['acc'])*100:.1f}pp", "LR leads accuracy")
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

col1,col2 = st.columns(2)

with col1:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>ROC Curves</div>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rf['fpr'],y=rf['tpr'],mode='lines',
        name=f"Random Forest (AUC {rf['auc']:.3f})",
        line=dict(color=COLORS['blue'],width=3),fill='tozeroy',fillcolor='rgba(59,130,246,0.06)',
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra>RF</extra>'))
    fig.add_trace(go.Scatter(x=lr['fpr'],y=lr['tpr'],mode='lines',
        name=f"Logistic Regression (AUC {lr['auc']:.3f})",
        line=dict(color=COLORS['green'],width=3,dash='dash'),
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra>LR</extra>'))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Baseline (AUC 0.500)',
        line=dict(color=COLORS['muted'],width=1.5,dash='dot')))
    theme(fig,'ROC Curves — WTP Prediction',380,margin=dict(l=0,r=0,t=40,b=0),
          xaxis_title='False Positive Rate',yaxis_title='True Positive Rate',
          legend=dict(x=0.4,y=0.12,font=dict(size=11),bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    idx = np.argmin(np.abs(np.array(lr['fpr'])-0.3))
    st.markdown(f"""<div class="insight-card"><div class="label">What This Means for Growth</div>
    If you ran a targeted upgrade campaign to 200 users, this model would correctly identify
    <b>~{int(200*lr['tpr'][idx])} genuine high-paying users</b> — without requiring any manual
    review. That is the difference between <b>spray-and-pray marketing</b> and a precision
    conversion engine. At a {wtp_med:.0f}/mo price point, those {int(200*lr['tpr'][idx])} conversions
    represent <b>${int(200*lr['tpr'][idx]*wtp_med*12):,} in annualised revenue</b> from a single campaign.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>Confusion Matrix — {model_choice}</div>", unsafe_allow_html=True)
    cm = active['cm']
    labels = ['Low WTP','High WTP']
    fig = go.Figure(go.Heatmap(
        z=cm,x=labels,y=labels,
        colorscale=[[0,'#0F2044'],[0.5,'#1D4ED8'],[1,'#3B82F6']],
        showscale=False,hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'))
    for i in range(2):
        for j in range(2):
            fig.add_annotation(x=labels[j],y=labels[i],text=f"<b>{cm[i,j]}</b>",
                showarrow=False,font=dict(size=28,family='JetBrains Mono',color='white'))
    theme(fig,f'{model_choice} — Accuracy {active["acc"]*100:.1f}%',300,
          margin=dict(l=60,r=10,t=40,b=60),xaxis_title='Predicted',yaxis_title='Actual')
    st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    tp,fn,fp,tn = cm[1,1],cm[1,0],cm[0,1],cm[0,0]
    st.markdown(f"""<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px;">
        <div style="background:#064E3B;border-radius:8px;padding:10px;text-align:center;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.6rem;color:#34D399;font-weight:700;">{tp}</div>
            <div style="font-size:0.72rem;color:#6EE7B7;">Revenue captured</div>
        </div>
        <div style="background:#450A0A;border-radius:8px;padding:10px;text-align:center;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.6rem;color:#F87171;font-weight:700;">{fn}</div>
            <div style="font-size:0.72rem;color:#FCA5A5;">Missed revenue</div>
        </div></div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

col1,col2 = st.columns(2)

with col1:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>Feature Importance — Payment Predictors</div>", unsafe_allow_html=True)
    top_n = st.slider("Features",6,18,10,label_visibility='collapsed')
    imp = res['importances'].head(top_n).reset_index()
    imp.columns=['Feature','Importance']
    imp['Feature']=imp['Feature'].str.replace('_',' ').str.replace(' enc','')
    imp_s=imp.sort_values('Importance')
    bar_clrs=[COLORS['amber'] if i>=len(imp_s)-3 else COLORS['blue'] for i in range(len(imp_s))]
    fig=go.Figure(go.Bar(y=imp_s['Feature'],x=imp_s['Importance'],orientation='h',
        marker_color=bar_clrs,marker_opacity=0.9,
        text=[f"{v:.3f}" for v in imp_s['Importance']],textposition='outside',
        textfont=dict(size=10,family='JetBrains Mono',color=COLORS['muted']),
        hovertemplate='%{y}: %{x:.4f}<extra></extra>'))
    theme(fig,f'RF Feature Importance (Top {top_n})',max(320,top_n*32),
          margin=dict(l=0,r=70,t=40,b=0),xaxis_title='Gini Importance')
    st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    top3=res['importances'].head(3).index.str.replace('_',' ').tolist()
    st.markdown(f"""<div class="insight-card"><div class="label">What This Means for Pricing Strategy</div>
    The top payment predictors are <b>outcome metrics, not demographics</b>: {top3[0]}, {top3[1]},
    and {top3[2]}. Students do not pay because they are in Year 3 or studying Engineering.
    <b>They pay when the product has already worked for them.</b> This is your pricing strategy
    in one sentence: show results first, charge second. A 14-day free trial is not a cost —
    it is a conversion mechanism. Every student who sees a GPA improvement during their trial
    is a paying subscriber waiting to happen.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>WTP Distribution by Segment</div>", unsafe_allow_html=True)
    df_wtp = res['df'][res['df']['WTP_High']==1] if show_high_only else res['df']
    fig = go.Figure()
    for persona in df_wtp.groupby('Persona')['Willingness_to_Pay'].median().sort_values(ascending=False).index:
        sub   = df_wtp[df_wtp['Persona']==persona]['Willingness_to_Pay']
        color = PERSONA_PALETTE.get(persona,'#888')
        r,g,b = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
        fig.add_trace(go.Violin(y=sub,name=persona,box_visible=True,meanline_visible=True,
            fillcolor=f'rgba({r},{g},{b},0.15)',line_color=color,
            meanline=dict(color='white',width=2),points='outliers',
            marker=dict(color=color,size=3,opacity=0.5),
            hovertemplate=f'<b>{persona}</b><br>WTP: $%{{y:.0f}}/mo<extra></extra>'))
    fig.add_hline(y=wtp_med,line_color=COLORS['amber'],line_dash='dash',line_width=2,
                   annotation_text=f"Median ${wtp_med:.0f}",annotation_font_color=COLORS['amber'])
    theme(fig,'WTP by Segment'+('' if not show_high_only else ' (High WTP only)'),380,
          margin=dict(l=0,r=10,t=40,b=0),yaxis_title='Willingness to Pay ($/mo)')
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    st.markdown("""<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:4px;">
        <div style="background:#111827;border:1px solid #1E2D45;border-top:2px solid #EF4444;border-radius:8px;padding:10px;text-align:center;">
            <div style="font-size:0.65rem;color:#64748B;text-transform:uppercase;">Entry</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;color:#F87171;font-weight:700;">$35/mo</div>
            <div style="font-size:0.7rem;color:#475569;">Developing Strivers</div>
        </div>
        <div style="background:#111827;border:1px solid #1E2D45;border-top:2px solid #3B82F6;border-radius:8px;padding:10px;text-align:center;">
            <div style="font-size:0.65rem;color:#64748B;text-transform:uppercase;">Core</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;color:#60A5FA;font-weight:700;">$45/mo</div>
            <div style="font-size:0.7rem;color:#475569;">High-Stress Engagers</div>
        </div>
        <div style="background:#111827;border:1px solid #1E2D45;border-top:2px solid #10B981;border-radius:8px;padding:10px;text-align:center;">
            <div style="font-size:0.65rem;color:#64748B;text-transform:uppercase;">Premium</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;color:#34D399;font-weight:700;">$55/mo</div>
            <div style="font-size:0.7rem;color:#475569;">Efficient Achievers</div>
        </div></div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
