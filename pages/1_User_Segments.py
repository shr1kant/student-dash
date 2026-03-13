"""pages/1_User_Segments.py — ALO Platform · User Segmentation"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_embedded import load_raw
from engine import prepare_data, run_clustering, COLORS, GLOBAL_CSS, PERSONA_PALETTE, theme

st.set_page_config(page_title="User Segments · ALO", page_icon="🎯", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div style='font-size:1rem;font-weight:700;color:#E2E8F0;padding:16px 0 12px;border-bottom:1px solid #1E2D45;margin-bottom:16px;'>User Segments</div>", unsafe_allow_html=True)
    k_val = st.slider("Number of segments (k)", 2, 8, 4)
    drill_major = st.selectbox("Drill down by Major", ["All","Business","Computer Science","Data Science","Engineering","Finance","Healthcare","Law","Marketing"])
    st.markdown("---")
    st.caption("K-Means on 9 behavioural signals. k=4 validated by silhouette score.")

@st.cache_data
def get_segments(k):
    df = prepare_data(load_raw())
    return run_clustering(df, k=k)

df_c, info = get_segments(k_val)
df_drill = df_c[df_c['Major']==drill_major] if (drill_major!="All" and drill_major in df_c['Major'].unique()) else df_c

st.markdown(f"""<div style="padding:0 0 24px 0;border-bottom:1px solid #1E2D45;margin-bottom:24px;">
    <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.12em;color:#8B5CF6;font-weight:600;margin-bottom:6px;">User Segmentation</div>
    <h1 style="margin:0;font-size:1.8rem;font-weight:700;color:#E2E8F0;">Who Are Our Users?</h1>
    <p style="margin:6px 0 0 0;color:#64748B;font-size:0.88rem;">K-Means · {k_val} segments · n={len(df_drill)} users</p>
</div>""", unsafe_allow_html=True)

personas = df_drill['Persona'].value_counts()
cols = st.columns(len(personas)+2)
for i,(persona,count) in enumerate(personas.items()):
    color = PERSONA_PALETTE.get(persona,'#888')
    pct = count/len(df_drill)*100
    cols[i].markdown(f"""<div style="background:#111827;border:1px solid #1E2D45;border-top:3px solid {color};
         border-radius:10px;padding:14px;text-align:center;">
        <div style="font-size:0.65rem;color:#64748B;text-transform:uppercase;margin-bottom:4px;">{persona}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.6rem;font-weight:700;color:{color};">{pct:.0f}%</div>
        <div style="font-size:0.75rem;color:#475569;">n = {count}</div>
    </div>""", unsafe_allow_html=True)
cols[-2].metric("Silhouette", f"{info['best_silhouette']:.3f}")
cols[-1].metric("PCA Variance", f"{sum(info['pca_var'][:2])*100:.0f}%")
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

col1, col2 = st.columns([6,4])

with col1:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>Behavioural Cluster Map</div>", unsafe_allow_html=True)
    fig = go.Figure()
    for persona in df_drill['Persona'].unique():
        mask = df_drill['Persona']==persona
        sub  = df_drill[mask]
        color = PERSONA_PALETTE.get(persona,'#888')
        fig.add_trace(go.Scatter(
            x=sub['PCA1'],y=sub['PCA2'],mode='markers',
            name=f"{persona} ({mask.sum()})",
            marker=dict(color=color,size=7,opacity=0.7,line=dict(width=0.8,color=COLORS['bg'])),
            hovertemplate=f"<b>{persona}</b><br>GPA: %{{customdata[0]:.2f}}<br>Engagement: %{{customdata[1]:.1f}}<br>Stress: %{{customdata[2]:.1f}}<br>WTP: $%{{customdata[3]:.0f}}<extra></extra>",
            customdata=sub[['GPA_Change','Engagement_Level','Stress_Level','Willingness_to_Pay']].values))
        fig.add_trace(go.Scatter(
            x=[sub['PCA1'].mean()],y=[sub['PCA2'].mean()],mode='markers',
            marker=dict(symbol='x',size=16,color='white',line=dict(width=2.5,color=color)),
            showlegend=False,hoverinfo='skip'))
    theme(fig,f'PCA Projection — {info["pca_var"][0]*100:.0f}% + {info["pca_var"][1]*100:.0f}% variance',420,
          margin=dict(l=0,r=0,t=40,b=60),
          xaxis_title=f'PC1 ({info["pca_var"][0]*100:.0f}%)',
          yaxis_title=f'PC2 ({info["pca_var"][1]*100:.0f}%)',
          legend=dict(orientation='h',y=-0.15,x=0,font=dict(size=10),bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    st.markdown("""<div class="insight-card"><div class="label">What This Means for the Business</div>
    Each coloured cluster is a <b>distinct type of student with different needs, different pain points,
    and different price sensitivity</b>. Efficient Achievers (tight cluster) are a small but
    predictable group — easy to serve, easy to upsell. High-Stress Engagers (spread wide) are your
    largest segment and your biggest opportunity: they are already using the platform heavily,
    they just need help converting that engagement into visible results.
    <b>Hover over individual dots to inspect any student's full profile.</b>
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>Persona Fingerprints</div>", unsafe_allow_html=True)
    radar_feats = ['Average_Test_Score','Stress_Level','Engagement_Level','Study_Efficiency_Index','GPA_Change','Satisfaction_Score']
    radar_lbls  = ['Test Score','Stress','Engagement','Efficiency','GPA','Satisfaction']
    cm = df_drill.groupby('Persona')[radar_feats].mean()
    cn = (cm-cm.min())/(cm.max()-cm.min())
    def hex_rgba(h,a=0.12):
        h=h.lstrip('#'); r,g,b=int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
        return f'rgba({r},{g},{b},{a})'
    fig = go.Figure()
    for persona in cn.index:
        color = PERSONA_PALETTE.get(persona,'#888')
        vals  = cn.loc[persona].tolist()+[cn.loc[persona].tolist()[0]]
        lbls  = radar_lbls+[radar_lbls[0]]
        fig.add_trace(go.Scatterpolar(r=vals,theta=lbls,fill='toself',
            fillcolor=hex_rgba(color),line=dict(color=color,width=2.5),name=persona))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        polar=dict(bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True,range=[0,1],showticklabels=False,
                            gridcolor='#1E293B',linecolor='#1E293B'),
            angularaxis=dict(gridcolor='#1E293B',linecolor='#1E293B',
                             tickfont=dict(color=COLORS['muted'],size=10))),
        showlegend=True,height=420,margin=dict(l=20,r=20,t=20,b=60),
        font=dict(color=COLORS['text'],family='DM Sans'),
        legend=dict(orientation='h',y=-0.15,x=0,font=dict(size=9,color=COLORS['muted']),bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    st.markdown("</div>", unsafe_allow_html=True)

col1, col2 = st.columns([4,6])

with col1:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>Optimal k Validation</div>", unsafe_allow_html=True)
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,
                         subplot_titles=('Inertia (Elbow)','Silhouette Score'),vertical_spacing=0.15)
    fig.add_trace(go.Scatter(x=info['k_range'],y=info['inertias'],mode='lines+markers',
        line=dict(color=COLORS['blue'],width=3),marker=dict(size=8),name='Inertia',
        hovertemplate='k=%{x}: %{y:.0f}<extra></extra>'),row=1,col=1)
    fig.add_trace(go.Scatter(x=info['k_range'],y=info['silhouettes'],mode='lines+markers',
        line=dict(color=COLORS['green'],width=3),marker=dict(size=8),name='Silhouette',
        hovertemplate='k=%{x}: %{y:.3f}<extra></extra>'),row=2,col=1)
    for r in [1,2]:
        fig.add_vline(x=k_val,line_color=COLORS['amber'],line_dash='dash',line_width=1.5,row=r,col=1)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text'],family='DM Sans'),height=320,showlegend=False,
        margin=dict(l=0,r=10,t=40,b=10))
    for ax in ['xaxis','xaxis2','yaxis','yaxis2']:
        fig.update_layout(**{ax:dict(gridcolor=COLORS['faint'],linecolor=COLORS['border'])})
    st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    st.markdown(f"""<div class="insight-card"><div class="label">Why This Number of Segments?</div>
    The silhouette score of <b>{info['best_silhouette']:.3f}</b> tells us that these {k_val} groups
    are genuinely different from each other — not just randomly drawn lines in the data.
    Think of it as a "how distinct are these groups?" score: above 0.15 means the segments
    are real enough to build separate marketing, pricing, and product strategies around.
    Use the sidebar to experiment — but k=4 is where the data naturally wants to split.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>Segment Deep-Dive</div>", unsafe_allow_html=True)
    metric_map = {'GPA Improvement':('GPA_Change','GPA Change'),
                  'Willingness to Pay':('Willingness_to_Pay','WTP ($/mo)'),
                  'Satisfaction':('Satisfaction_Score','Score /10'),
                  'Engagement':('Engagement_Level','Score /10'),
                  'Stress Level':('Stress_Level','Score /10'),
                  'Study Efficiency':('Study_Efficiency_Index','Score /10')}
    mlabel = st.selectbox("Metric",list(metric_map.keys()),label_visibility='collapsed')
    mcol,maxis = metric_map[mlabel]
    fig = go.Figure()
    for persona in df_drill.groupby('Persona')[mcol].median().sort_values().index:
        sub   = df_drill[df_drill['Persona']==persona][mcol]
        color = PERSONA_PALETTE.get(persona,'#888')
        r,g,b = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
        fig.add_trace(go.Box(y=sub,name=persona,marker_color=color,line_color=color,
            fillcolor=f'rgba({r},{g},{b},0.15)',boxpoints='outliers',
            hovertemplate=f'<b>{persona}</b><br>{maxis}: %{{y:.2f}}<extra></extra>'))
    avg=df_drill[mcol].mean()
    fig.add_hline(y=avg,line_color=COLORS['muted'],line_dash='dot',line_width=1.5,
                   annotation_text=f"Avg {avg:.2f}",annotation_font_color=COLORS['muted'],annotation_font_size=10)
    theme(fig,f'{mlabel} by Segment',320,margin=dict(l=0,r=20,t=40,b=0),yaxis_title=maxis)
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    best  = df_drill.groupby('Persona')[mcol].median().idxmax()
    worst = df_drill.groupby('Persona')[mcol].median().idxmin()
    bv    = df_drill.groupby('Persona')[mcol].median().max()
    wv    = df_drill.groupby('Persona')[mcol].median().min()
    st.markdown(f"""<div class="insight-card"><div class="label">What This Gap Costs You</div>
    <b>{best}</b> is at {bv:.2f}. <b>{worst}</b> is at {wv:.2f}. That gap of <b>{bv-wv:.2f}</b>
    is not fixed — it is a product problem with a product solution. The question is not
    "why is one group better?" but <b>"what feature, nudge, or intervention would close this gap
    by even 20%?"</b> A 20% lift in the bottom segment, at scale, has more revenue impact than
    any optimisation you could make for your top segment.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='alo-card' style='margin-top:8px;'>", unsafe_allow_html=True)
st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:16px;'>Segment Summary Table</div>", unsafe_allow_html=True)
summary = df_drill.groupby('Persona').agg(
    Users=('Persona','count'),Avg_GPA=('GPA_Change','mean'),
    Avg_WTP=('Willingness_to_Pay','mean'),Satisfaction=('Satisfaction_Score','mean'),
    Engagement=('Engagement_Level','mean'),Stress=('Stress_Level','mean')).round(2).reset_index()
summary.columns=['Segment','Users','Avg GPA','Avg WTP ($)','Satisfaction','Engagement','Stress']
st.dataframe(summary.style.background_gradient(subset=['Avg GPA'],cmap='RdYlGn')
                           .background_gradient(subset=['Avg WTP ($)'],cmap='Blues')
                           .background_gradient(subset=['Stress'],cmap='RdYlGn_r'),
             hide_index=True,use_container_width=True,height=200)
st.download_button("Download segment data",
    df_drill[['Student_ID','Major','Year_of_Study','Persona','GPA_Change','Willingness_to_Pay','Satisfaction_Score']].to_csv(index=False).encode(),
    "ALO_segments.csv","text/csv")
st.markdown("</div>", unsafe_allow_html=True)
