"""
pages/1_User_Segments.py — ALO Platform · User Segmentation Intelligence
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
from engine import prepare_data, run_clustering, COLORS, PLOTLY_LAYOUT, GLOBAL_CSS, PERSONA_PALETTE, apply_theme

st.set_page_config(page_title="User Segments · ALO", page_icon="◉", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="font-size:1rem;font-weight:700;color:#E2E8F0;padding:16px 0 12px;border-bottom:1px solid #1E2D45;margin-bottom:16px;">◉ User Segments</div>""", unsafe_allow_html=True)
    k_val = st.slider("Segments (k)", 2, 8, 4, help="Number of user clusters. k=4 is validated by silhouette score.")
    show_centroids = st.toggle("Show cluster centroids", True)
    drill_major    = st.selectbox("Drill down by Major", ["All"] + ["Business","Computer Science","Data Science","Engineering","Finance","Healthcare","Law","Marketing"])
    st.markdown("---")
    st.markdown("""<div style="font-size:0.72rem;color:#475569;">
    Clustering uses K-Means on 9 behavioural signals. Optimal k is validated by both
    Elbow Method and Silhouette Score.
    </div>""", unsafe_allow_html=True)

@st.cache_data
def get_segments(k):
    raw = load_raw()
    df  = prepare_data(raw)
    return run_clustering(df, k=k)

df_c, info = get_segments(k_val)
if drill_major != "All" and drill_major in df_c['Major'].unique():
    df_drill = df_c[df_c['Major'] == drill_major]
else:
    df_drill = df_c

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding:0 0 24px 0; border-bottom:1px solid #1E2D45; margin-bottom:24px;">
    <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; color:#8B5CF6; font-weight:600; margin-bottom:6px;">User Segmentation</div>
    <h1 style="margin:0; font-size:1.8rem; font-weight:700; color:#E2E8F0; letter-spacing:-0.02em;">Who Are Our Users?</h1>
    <p style="margin:6px 0 0 0; color:#64748B; font-size:0.88rem;">K-Means clustering · {k_val} behavioural segments · n={len(df_drill)} users {f'(filtered: {drill_major})' if drill_major != 'All' else ''}</p>
</div>
""", unsafe_allow_html=True)

# ── Segment KPI pills ─────────────────────────────────────────────────────────
personas = df_drill['Persona'].value_counts()
cols = st.columns(len(personas) + 2)
for i, (persona, count) in enumerate(personas.items()):
    color = PERSONA_PALETTE.get(persona, '#888')
    pct = count/len(df_drill)*100
    cols[i].markdown(f"""
    <div style="background:#111827;border:1px solid #1E2D45;border-top:3px solid {color};
         border-radius:10px;padding:14px 16px;text-align:center;">
        <div style="font-size:0.68rem;color:#64748B;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">{persona}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.6rem;font-weight:700;color:{color};">{pct:.0f}%</div>
        <div style="font-size:0.75rem;color:#475569;margin-top:2px;">n = {count}</div>
    </div>""", unsafe_allow_html=True)

cols[-2].metric("Silhouette", f"{info['best_silhouette']:.3f}", "Cluster quality")
cols[-1].metric("PCA Variance", f"{sum(info['pca_var'][:2])*100:.0f}%", "Explained in 2D")

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── Row 1: PCA map + Radar ────────────────────────────────────────────────────
col1, col2 = st.columns([6,4])

with col1:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>Behavioural Cluster Map · Click to explore</div>", unsafe_allow_html=True)

    fig = go.Figure()
    for persona in df_drill['Persona'].unique():
        mask = df_drill['Persona'] == persona
        sub  = df_drill[mask]
        color = PERSONA_PALETTE.get(persona, '#888')
        fig.add_trace(go.Scatter(
            x=sub['PCA1'], y=sub['PCA2'],
            mode='markers',
            name=f"{persona} ({mask.sum()})",
            marker=dict(color=color, size=7, opacity=0.7,
                        line=dict(width=0.8, color=COLORS['bg'])),
            hovertemplate=(
                f"<b style='color:{color}'>{persona}</b><br>"
                "GPA Δ: %{customdata[0]:.2f}<br>"
                "Engagement: %{customdata[1]:.1f}/10<br>"
                "Stress: %{customdata[2]:.1f}/10<br>"
                "WTP: $%{customdata[3]:.0f}/mo<extra></extra>"
            ),
            customdata=sub[['GPA_Change','Engagement_Level','Stress_Level','Willingness_to_Pay']].values
        ))

    if show_centroids:
        for persona in df_drill['Persona'].unique():
            sub = df_drill[df_drill['Persona'] == persona]
            cx, cy = sub['PCA1'].mean(), sub['PCA2'].mean()
            color = PERSONA_PALETTE.get(persona, '#888')
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy], mode='markers',
                marker=dict(symbol='x', size=16, color='white',
                            line=dict(width=2.5, color=color)),
                showlegend=False, hoverinfo='skip'
            ))

    fig.update_layout(**PLOTLY_LAYOUT,
        title=f'PCA Projection — PC1 {info["pca_var"][0]*100:.0f}% · PC2 {info["pca_var"][1]*100:.0f}% variance',
        height=420,
        legend=dict(orientation='h', y=-0.12, x=0, font=dict(size=10)),
        xaxis_title=f'Principal Component 1 ({info["pca_var"][0]*100:.0f}%)',
        yaxis_title=f'Principal Component 2 ({info["pca_var"][1]*100:.0f}%)',
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("""<div class="insight-card">
    <div class="label">◈ Segmentation Insight</div>
    <b>Efficient Achievers cluster tightly</b> in the top-right of the PCA space — a rare but
    high-value cohort with distinct behavioural fingerprints. <b>High-Stress Engagers spread
    wide</b>, indicating internal variance ripe for sub-segmentation by major or year.
    Use this map to identify outlier users who don't fit any persona cleanly — they're often
    the most interesting conversion opportunities.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>Persona Fingerprints · Radar</div>", unsafe_allow_html=True)

    radar_feats = ['Average_Test_Score','Stress_Level','Engagement_Level',
                    'Study_Efficiency_Index','GPA_Change','Satisfaction_Score']
    radar_labels = ['Test Score','Stress','Engagement','Efficiency','GPA Δ','Satisfaction']

    cm = df_drill.groupby('Persona')[radar_feats].mean()
    cn = (cm - cm.min()) / (cm.max() - cm.min())

    fig = go.Figure()
    for persona in cn.index:
        color = PERSONA_PALETTE.get(persona, '#888')
        vals = cn.loc[persona].tolist() + [cn.loc[persona].tolist()[0]]
        lbls = radar_labels + [radar_labels[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=lbls,
            fill='toself', fillcolor=color.replace('#','rgba(').replace(')',',0.12)') if color.startswith('#') else color,
            line=dict(color=color, width=2.5),
            name=persona,
            hovertemplate='%{theta}: %{r:.2f}<extra>' + persona + '</extra>'
        ))

    # Approximate rgba for hex colors
    def hex_rgba(h, a=0.12):
        h = h.lstrip('#')
        r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f'rgba({r},{g},{b},{a})'

    fig.data = []
    for persona in cn.index:
        color = PERSONA_PALETTE.get(persona, '#888')
        vals = cn.loc[persona].tolist() + [cn.loc[persona].tolist()[0]]
        lbls = radar_labels + [radar_labels[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=lbls,
            fill='toself', fillcolor=hex_rgba(color),
            line=dict(color=color, width=2.5),
            name=persona,
        ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0,1], showticklabels=False,
                            gridcolor='#1E293B', linecolor='#1E293B'),
            angularaxis=dict(gridcolor='#1E293B', linecolor='#1E293B',
                             tickfont=dict(color=COLORS['muted'], size=10))
        ),
        showlegend=True,
        legend=dict(orientation='h', y=-0.15, x=0, font=dict(size=9, color=COLORS['muted']),
                    bgcolor='rgba(0,0,0,0)'),
        font=dict(color=COLORS['text'], family='DM Sans'),
        height=420, margin=dict(l=20,r=20,t=20,b=60),
        title=''
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

# ── Row 2: Elbow/Silhouette + Segment deep-dive ───────────────────────────────
col1, col2 = st.columns([4,6])

with col1:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>Optimal k Validation</div>", unsafe_allow_html=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=('Inertia (Elbow Method)', 'Silhouette Score'),
                         vertical_spacing=0.15)
    fig.add_trace(go.Scatter(x=info['k_range'], y=info['inertias'],
                              mode='lines+markers', line=dict(color=COLORS['blue'],width=3),
                              marker=dict(size=8), name='Inertia',
                              hovertemplate='k=%{x}: Inertia=%{y:.0f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=info['k_range'], y=info['silhouettes'],
                              mode='lines+markers', line=dict(color=COLORS['green'],width=3),
                              marker=dict(size=8), name='Silhouette',
                              hovertemplate='k=%{x}: Score=%{y:.3f}<extra></extra>'), row=2, col=1)
    for r in [1,2]:
        fig.add_vline(x=k_val, line_color=COLORS['amber'], line_dash='dash', line_width=1.5, row=r, col=1)

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       font=dict(color=COLORS['text'], family='DM Sans'),
                       height=320, showlegend=False, margin=dict(l=0,r=10,t=40,b=10))
    for ax in ['xaxis','xaxis2','yaxis','yaxis2']:
        fig.update_layout(**{ax: dict(gridcolor=COLORS['faint'], linecolor=COLORS['border'])})
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown(f"""<div class="insight-card">
    <div class="label">◈ Model Validation</div>
    k={k_val} gives the best silhouette score of <b>{info['best_silhouette']:.3f}</b>.
    Scores above 0.15 indicate meaningful structure in behavioural data.
    Try adjusting k in the sidebar to see how segments shift.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>Segment Deep-Dive · Select a metric</div>", unsafe_allow_html=True)

    metric_map = {
        'GPA Improvement':    ('GPA_Change', 'GPA Change'),
        'Willingness to Pay': ('Willingness_to_Pay', 'WTP ($/mo)'),
        'Satisfaction Score': ('Satisfaction_Score', 'Satisfaction /10'),
        'Engagement Level':   ('Engagement_Level', 'Engagement /10'),
        'Stress Level':       ('Stress_Level', 'Stress /10'),
        'Study Efficiency':   ('Study_Efficiency_Index', 'Efficiency /10'),
    }
    metric_label = st.selectbox("Metric", list(metric_map.keys()), label_visibility='collapsed')
    metric_col, metric_axis = metric_map[metric_label]

    personas_sorted = df_drill.groupby('Persona')[metric_col].median().sort_values().index.tolist()

    fig = go.Figure()
    for persona in personas_sorted:
        sub = df_drill[df_drill['Persona'] == persona][metric_col]
        color = PERSONA_PALETTE.get(persona, '#888')
        fig.add_trace(go.Box(
            y=sub, name=persona,
            marker_color=color, line_color=color,
            fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)',
            boxpoints='outliers', pointpos=0,
            marker=dict(size=4, opacity=0.5),
            hovertemplate=f'<b>{persona}</b><br>{metric_axis}: %{{y:.2f}}<extra></extra>'
        ))

    avg = df_drill[metric_col].mean()
    fig.add_hline(y=avg, line_color=COLORS['muted'], line_dash='dot', line_width=1.5,
                   annotation_text=f"Avg {avg:.2f}", annotation_font_color=COLORS['muted'],
                   annotation_font_size=10)

    fig.update_layout(**PLOTLY_LAYOUT, title=f'{metric_label} by Segment',
                       height=320, margin=dict(l=0,r=20,t=40,b=0),
                       yaxis_title=metric_axis,
                       xaxis=dict(showgrid=False))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Auto-insight based on selected metric
    best_seg = df_drill.groupby('Persona')[metric_col].median().idxmax()
    worst_seg = df_drill.groupby('Persona')[metric_col].median().idxmin()
    best_val = df_drill.groupby('Persona')[metric_col].median().max()
    worst_val = df_drill.groupby('Persona')[metric_col].median().min()

    st.markdown(f"""<div class="insight-card">
    <div class="label">◈ Auto Insight · {metric_label}</div>
    <b>{best_seg}</b> leads with a median of <b>{best_val:.2f}</b>.
    <b>{worst_seg}</b> trails at <b>{worst_val:.2f}</b> — a gap of <b>{best_val-worst_val:.2f}</b>
    ({((best_val-worst_val)/abs(worst_val)*100 if worst_val!=0 else 0):.0f}% difference).
    Closing this gap through targeted interventions is ALO's highest-leverage growth action.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Segment summary table ─────────────────────────────────────────────────────
st.markdown("<div class='alo-card' style='margin-top:8px;'>", unsafe_allow_html=True)
st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:16px;'>Segment Performance Summary</div>", unsafe_allow_html=True)

summary = df_drill.groupby('Persona').agg(
    Users=('Persona','count'),
    Avg_GPA=('GPA_Change','mean'),
    Avg_WTP=('Willingness_to_Pay','mean'),
    Avg_Satisfaction=('Satisfaction_Score','mean'),
    Avg_Engagement=('Engagement_Level','mean'),
    Avg_Stress=('Stress_Level','mean'),
    Avg_Efficiency=('Study_Efficiency_Index','mean'),
).round(2).reset_index()
summary.columns = ['Segment','Users','Avg GPA Δ','Avg WTP ($)','Satisfaction','Engagement','Stress','Efficiency']

st.dataframe(
    summary.style
        .background_gradient(subset=['Avg GPA Δ'], cmap='RdYlGn')
        .background_gradient(subset=['Avg WTP ($)'], cmap='Blues')
        .background_gradient(subset=['Stress'], cmap='RdYlGn_r'),
    hide_index=True, use_container_width=True, height=200
)
csv = df_drill[['Student_ID','Major','Year_of_Study','Persona','GPA_Change','Willingness_to_Pay','Satisfaction_Score','Engagement_Level']].to_csv(index=False).encode()
st.download_button("⬇ Export segment data", csv, "ALO_segments.csv", "text/csv", use_container_width=False)
st.markdown("</div>", unsafe_allow_html=True)
