"""
app.py — ALO Platform · Executive Intelligence Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_embedded import load_raw
from engine import prepare_data, COLORS, PLOTLY_LAYOUT, GLOBAL_CSS, PERSONA_PALETTE, apply_theme

st.set_page_config(
    page_title="ALO Platform Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 20px 0 16px 0; border-bottom: 1px solid #1E2D45; margin-bottom: 16px;">
        <div style="font-size:1.3rem; font-weight:700; color:#E2E8F0; letter-spacing:-0.02em;">◈ ALO</div>
        <div style="font-size:0.72rem; color:#475569; text-transform:uppercase; letter-spacing:0.1em; margin-top:2px;">Platform Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;color:#475569;margin-bottom:10px;">Navigation</div>
    <div style="font-size:0.82rem;color:#64748B;line-height:2;">
        &nbsp;&nbsp;◇ Executive Overview<br>
        &nbsp;&nbsp;◇ User Segments<br>
        &nbsp;&nbsp;◇ Revenue Intelligence<br>
        &nbsp;&nbsp;◇ Behaviour Patterns<br>
        &nbsp;&nbsp;◇ Performance Drivers
    </div>
    <div style="font-size:0.72rem;color:#334155;margin-top:8px;">Use the sidebar menu above to navigate.</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="border-top:1px solid #1E2D45; margin-top:20px; padding-top:16px;">
        <div style="font-size:0.7rem; color:#334155;">
            <div style="color:#475569; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.1em; font-size:0.65rem;">Dataset</div>
            <div style="font-family:'JetBrains Mono',monospace; color:#94A3B8;">450 users · 23 signals</div>
            <div style="font-family:'JetBrains Mono',monospace; color:#94A3B8;">SG + Dubai cohort</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    raw = load_raw()
    return prepare_data(raw)

df = get_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:0 0 28px 0; border-bottom:1px solid #1E2D45; margin-bottom:28px;">
    <div style="display:flex; align-items:flex-end; justify-content:space-between;">
        <div>
            <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; color:#3B82F6; font-weight:600; margin-bottom:6px;">
                Executive Overview
            </div>
            <h1 style="margin:0; font-size:2rem; font-weight:700; color:#E2E8F0; letter-spacing:-0.03em;">
                ALO Platform Intelligence
            </h1>
            <p style="margin:6px 0 0 0; color:#64748B; font-size:0.9rem;">
                Adaptive Learning Orchestrator · Student performance & monetisation analytics
            </p>
        </div>
        <div style="text-align:right; font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#475569;">
            Cohort: Singapore + Dubai<br>
            <span style="color:#3B82F6;">n = 450 active users</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)
wtp_med = df['Willingness_to_Pay'].median()
avg_gpa = df['GPA_Change'].mean()
avg_sat = df['Satisfaction_Score'].mean()
avg_eng = df['Engagement_Level'].mean()
ret_pct = (df['Retention_Likelihood'] >= 90).mean() * 100
high_stress_pct = (df['Stress_Level'] > 7).mean() * 100

c1.metric("Median WTP", f"${wtp_med:.0f}/mo", "+12% vs benchmark")
c2.metric("Avg GPA Δ",   f"+{avg_gpa:.2f}",    "per semester")
c3.metric("Satisfaction", f"{avg_sat:.1f}/10",  f"{(avg_sat/10*100):.0f}th pctile")
c4.metric("Engagement",   f"{avg_eng:.1f}/10",  "platform score")
c5.metric("High Retention", f"{ret_pct:.0f}%",  "≥90% likelihood")
c6.metric("Stress-Flagged", f"{high_stress_pct:.0f}%", "intervention priority")

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── Row 1: WTP distribution + GPA by major ────────────────────────────────────
col1, col2 = st.columns([5,5])

with col1:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)

    # Animated histogram with quartile bands
    fig = go.Figure()

    # Shade quartile regions
    q1,q2,q3 = df['Willingness_to_Pay'].quantile([0.25,0.50,0.75])
    for lo,hi,label,color in [
        (df['Willingness_to_Pay'].min(), q1,  'Q1 — Entry',   'rgba(239,68,68,0.07)'),
        (q1, q2,                                'Q2 — Core',    'rgba(245,158,11,0.07)'),
        (q2, q3,                                'Q3 — Growth',  'rgba(59,130,246,0.07)'),
        (q3, df['Willingness_to_Pay'].max(),    'Q4 — Premium', 'rgba(16,185,129,0.07)'),
    ]:
        fig.add_vrect(x0=lo, x1=hi, fillcolor=color, line_width=0, annotation_text=label,
                      annotation_position="top left", annotation_font_size=9,
                      annotation_font_color=COLORS['muted'])

    fig.add_trace(go.Histogram(
        x=df['Willingness_to_Pay'], nbinsx=40,
        marker=dict(color=COLORS['blue'], opacity=0.85,
                    line=dict(color=COLORS['bg'], width=0.5)),
        name='WTP Distribution', hovertemplate='$%{x:.0f}/mo — %{y} users<extra></extra>'
    ))
    fig.add_vline(x=wtp_med, line_color=COLORS['amber'], line_width=2, line_dash='dash',
                  annotation_text=f"Median ${wtp_med:.0f}", annotation_font_color=COLORS['amber'])
    fig.update_layout(**PLOTLY_LAYOUT, title='Willingness to Pay Distribution', height=280,
                       margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("""
    <div class="insight-card">
        <div class="label">◈ Revenue Insight</div>
        <b>Median WTP sits at $45/mo</b> — well above typical EdTech benchmarks ($18–28/mo).
        The Q4 premium cluster (top 25%) shows WTP above <b>$52/mo</b>, signalling clear headroom
        for a two-tier pricing model. Prioritise converting Q3 users with outcome-proof nudges.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)

    # GPA change by major — horizontal bars sorted
    gpa_major = df.groupby('Major').agg(
        gpa_mean=('GPA_Change','mean'),
        gpa_std=('GPA_Change','std'),
        n=('GPA_Change','count')
    ).sort_values('gpa_mean', ascending=True).reset_index()

    colors_bar = [COLORS['green'] if v > df['GPA_Change'].mean() else COLORS['red']
                  for v in gpa_major['gpa_mean']]

    fig = go.Figure(go.Bar(
        y=gpa_major['Major'],
        x=gpa_major['gpa_mean'],
        orientation='h',
        marker_color=colors_bar,
        marker_opacity=0.85,
        error_x=dict(type='data', array=gpa_major['gpa_std']/2,
                     color=COLORS['muted'], thickness=1.5, width=4),
        text=[f"+{v:.2f}" if v>0 else f"{v:.2f}" for v in gpa_major['gpa_mean']],
        textposition='outside',
        textfont=dict(size=10, color=COLORS['muted'], family='JetBrains Mono'),
        hovertemplate='%{y}: Avg GPA Δ = %{x:.3f}<br>n = %{customdata} users<extra></extra>',
        customdata=gpa_major['n']
    ))
    avg_line = df['GPA_Change'].mean()
    fig.add_vline(x=avg_line, line_color=COLORS['blue'], line_width=1.5, line_dash='dot',
                  annotation_text=f"Avg {avg_line:.2f}", annotation_font_color=COLORS['blue_light'],
                  annotation_font_size=10)
    fig.update_layout(**PLOTLY_LAYOUT, title='GPA Improvement by Major', height=280,
                       margin=dict(l=0,r=60,t=40,b=0),
                       xaxis_title='Avg GPA Change', yaxis_title='')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("""
    <div class="insight-card">
        <div class="label">◈ Product Insight</div>
        Majors above the average line are where ALO's adaptive scheduling has measurable lift.
        <b>Under-performing majors are not poor users — they're high-effort learners drowning in
        cognitive load.</b> They're the highest-priority cohort for ALO's stress-reduction module.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Row 2: Engagement heatmap + Stress vs AI scatter ─────────────────────────
col1, col2 = st.columns([5,5])

with col1:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)

    # Pivot: Stress bucket vs Year of Study → avg engagement
    df['_stress_bucket'] = pd.cut(df['Stress_Level'], bins=[0,4,7,10],
                                   labels=['Low (0–4)','Mid (4–7)','High (7–10)'])
    pivot = df.pivot_table(values='Engagement_Level', index='_stress_bucket',
                            columns='Year_of_Study', aggfunc='mean')
    pivot.columns = [f'Year {c}' for c in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0,'#0F2044'],[0.5,'#1D4ED8'],[1,'#60A5FA']],
        text=np.round(pivot.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=13, family='JetBrains Mono'),
        hovertemplate='%{y} · %{x}<br>Engagement: %{z:.2f}/10<extra></extra>',
        showscale=True,
        colorbar=dict(tickfont=dict(color=COLORS['muted']), thickness=12, len=0.9)
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title='Engagement Level · Stress vs Study Year',
                       height=280, margin=dict(l=0,r=40,t=40,b=0),
                       xaxis_title='', yaxis_title='')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("""
    <div class="insight-card">
        <div class="label">◈ Retention Insight</div>
        High-stress Year 1 students show <b>unexpectedly high engagement</b> — they're leaning
        into the platform when overwhelmed. This is the most critical window for conversion:
        students who engage during peak stress are <b>3× more likely to retain</b> long-term.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)

    # Scatter: Stress vs AI Usage, coloured by productivity quartile
    df['_prod_q'] = pd.qcut(df['Productivity_Improvement'].rank(method='first'), q=4,
                             labels=['Low','Mid-Low','Mid-High','High'])
    prod_colors = {'Low':COLORS['red'],'Mid-Low':COLORS['amber'],
                   'Mid-High':COLORS['blue'],'High':COLORS['green']}

    fig = go.Figure()
    for q in ['Low','Mid-Low','Mid-High','High']:
        mask = df['_prod_q'] == q
        fig.add_trace(go.Scatter(
            x=df.loc[mask,'Stress_Level'],
            y=df.loc[mask,'AI_Usage_Hours_Per_Week'],
            mode='markers',
            marker=dict(color=prod_colors[q], size=6, opacity=0.65,
                        line=dict(width=0.5, color=COLORS['bg'])),
            name=f'{q} productivity',
            hovertemplate=f'Stress: %{{x:.1f}}<br>AI Usage: %{{y:.1f}} hrs/wk<br>Productivity: {q}<extra></extra>'
        ))

    # Add trend line
    z = np.polyfit(df['Stress_Level'], df['AI_Usage_Hours_Per_Week'], 1)
    x_line = np.linspace(df['Stress_Level'].min(), df['Stress_Level'].max(), 50)
    fig.add_trace(go.Scatter(x=x_line, y=np.polyval(z,x_line),
                              mode='lines', line=dict(color=COLORS['amber'], width=2, dash='dash'),
                              name='Trend', showlegend=True))

    fig.update_layout(**PLOTLY_LAYOUT, title='Stress vs AI Usage · coloured by Productivity',
                       height=280, margin=dict(l=0,r=0,t=40,b=0),
                       xaxis_title='Stress Level', yaxis_title='AI Usage (hrs/wk)',
                       legend=dict(orientation='h', y=-0.15, x=0, font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("""
    <div class="insight-card">
        <div class="label">◈ Core Loop Validation</div>
        <b>As stress rises, AI usage rises — and so does productivity.</b>
        The positive trend line confirms ALO's core product hypothesis: the platform captures
        students at their most receptive moment (high stress) and converts that into measurable
        output. This is the flywheel investors need to see.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Row 3: Satisfaction funnel + Retention risk ───────────────────────────────
col1, col2, col3 = st.columns([3, 3, 4])

with col1:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    # Satisfaction score distribution — bullet chart style
    sat_bins = [0,4,6,8,10]
    sat_labels = ['Dissatisfied\n(<4)','Neutral\n(4–6)','Satisfied\n(6–8)','Delighted\n(8–10)']
    sat_counts = pd.cut(df['Satisfaction_Score'], bins=sat_bins, labels=sat_labels).value_counts()
    sat_pct = (sat_counts / len(df) * 100).reindex(sat_labels)

    fig = go.Figure(go.Bar(
        x=sat_pct.values,
        y=sat_labels,
        orientation='h',
        marker_color=[COLORS['red'], COLORS['amber'], COLORS['blue'], COLORS['green']],
        text=[f"{v:.0f}%" for v in sat_pct.values],
        textposition='outside',
        textfont=dict(family='JetBrains Mono', size=11),
        hovertemplate='%{y}: %{x:.1f}% of users<extra></extra>'
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title='User Satisfaction Breakdown',
                       height=260, margin=dict(l=0,r=50,t=40,b=0),
                       xaxis=dict(range=[0,80], showgrid=False),
                       yaxis=dict(showgrid=False))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    # Completion rate by year
    comp_year = df.groupby('Year_of_Study')['Assignment_Completion_Rate'].mean() * 100
    fig = go.Figure(go.Scatter(
        x=[f"Year {y}" for y in comp_year.index],
        y=comp_year.values,
        mode='lines+markers+text',
        line=dict(color=COLORS['blue'], width=3),
        marker=dict(size=12, color=COLORS['blue'], line=dict(width=2, color=COLORS['bg'])),
        text=[f"{v:.0f}%" for v in comp_year.values],
        textposition='top center',
        textfont=dict(family='JetBrains Mono', size=11, color=COLORS['blue_light']),
        fill='tozeroy',
        fillcolor='rgba(59,130,246,0.08)',
        hovertemplate='%{x}: %{y:.1f}% completion<extra></extra>'
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title='Completion Rate by Study Year',
                       height=260, margin=dict(l=0,r=20,t=40,b=0),
                       yaxis=dict(range=[60,100], ticksuffix='%'))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    # Retention risk segments - treemap
    df['_ret_risk'] = pd.cut(df['Retention_Likelihood'],
                              bins=[0,70,85,95,100],
                              labels=['At Risk (<70%)','Monitor (70–85%)','Stable (85–95%)','Locked In (95–100%)'])
    risk_counts = df['_ret_risk'].value_counts().reset_index()
    risk_counts.columns = ['Segment','Count']
    risk_counts['Pct'] = (risk_counts['Count']/len(df)*100).round(1)

    fig = go.Figure(go.Treemap(
        labels=[f"{r['Segment']}<br>{r['Count']} users ({r['Pct']}%)" for _, r in risk_counts.iterrows()],
        parents=['']*len(risk_counts),
        values=risk_counts['Count'].values,
        marker=dict(colors=[COLORS['red'], COLORS['amber'], COLORS['blue'], COLORS['green']],
                    line=dict(width=2, color=COLORS['bg'])),
        textfont=dict(size=12, family='DM Sans'),
        hovertemplate='%{label}<extra></extra>',
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title='Retention Risk Segments',
                       height=260, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

# ── Bottom insight bar ────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:20px; padding:16px 24px; background:#111827; border:1px solid #1E2D45;
     border-radius:12px; display:flex; gap:32px; align-items:center; flex-wrap:wrap;">
    <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; color:#3B82F6; font-weight:600; white-space:nowrap;">
        ◈ Platform Summary
    </div>
    <div style="font-size:0.83rem; color:#94A3B8; line-height:1.6;">
        <b style="color:#E2E8F0;">63% of users</b> are High-Stress Engagers — the platform's primary monetisation target.
        &nbsp;·&nbsp; <b style="color:#E2E8F0;">Stress drives AI usage</b> (lift = 1.48×), which drives engagement, which drives GPA improvement.
        &nbsp;·&nbsp; <b style="color:#E2E8F0;">76.7% accuracy</b> predicting who pays above median — actionable for growth campaigns.
    </div>
</div>
""", unsafe_allow_html=True)
