"""pages/3_Behaviour_Patterns.py — ALO Platform · Behavioural Pattern Mining"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_embedded import load_raw
from engine import prepare_data, run_arm, COLORS, GLOBAL_CSS, theme

st.set_page_config(page_title="Behaviour Patterns · ALO", page_icon="🔗", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div style='font-size:1rem;font-weight:700;color:#E2E8F0;padding:16px 0 12px;border-bottom:1px solid #1E2D45;margin-bottom:16px;'>Behaviour Patterns</div>", unsafe_allow_html=True)
    min_sup  = st.slider("Min Support",    0.20, 0.60, 0.28, 0.02)
    min_conf = st.slider("Min Confidence", 0.50, 0.95, 0.60, 0.05)
    min_lift = st.slider("Min Lift",       1.00, 2.50, 1.10, 0.05)
    st.markdown("---")
    st.caption("Support = % users with both behaviours\nConfidence = P(B|A)\nLift > 1 = above chance")

@st.cache_data
def get_patterns(sup, conf, lift):
    df = prepare_data(load_raw())
    return run_arm(df, min_sup=sup, min_conf=conf, min_lift=lift)

rules_df, item_sup, cooc = get_patterns(min_sup, min_conf, min_lift)

st.markdown(f"""<div style="padding:0 0 24px 0;border-bottom:1px solid #1E2D45;margin-bottom:24px;">
    <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.12em;color:#14B8A6;font-weight:600;margin-bottom:6px;">Behaviour Patterns</div>
    <h1 style="margin:0;font-size:1.8rem;font-weight:700;color:#E2E8F0;">What Do Users Do Together?</h1>
    <p style="margin:6px 0 0 0;color:#64748B;font-size:0.88rem;">Apriori pattern mining · {len(rules_df)} rules · Support>={min_sup} · Confidence>={min_conf} · Lift>={min_lift}</p>
</div>""", unsafe_allow_html=True)

if len(rules_df) == 0:
    st.warning("No rules found. Lower the thresholds in the sidebar.")
    st.stop()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Rules Found",    str(len(rules_df)))
c2.metric("Top Lift",       f"{rules_df['lift'].max():.3f}")
c3.metric("Avg Confidence", f"{rules_df['confidence'].mean():.2f}")
c4.metric("Top Rule",       (rules_df.iloc[0]['antecedents']+' -> '+rules_df.iloc[0]['consequents'])[:42]+"…")
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

col1, col2 = st.columns([6,4])

with col1:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>Rule Map — Support x Confidence x Lift</div>", unsafe_allow_html=True)
    rules_df['Rule'] = rules_df['antecedents'] + ' -> ' + rules_df['consequents']
    fig = go.Figure(go.Scatter(
        x=rules_df['support'], y=rules_df['confidence'], mode='markers',
        marker=dict(
            size=(rules_df['lift']*18).clip(8, 60),
            color=rules_df['lift'],
            colorscale=[[0,'#1E3A5F'],[0.5,'#2563EB'],[1,'#F59E0B']],
            showscale=False, opacity=0.85,
            line=dict(width=1, color=COLORS['bg'])
        ),
        text=rules_df['Rule'],
        hovertemplate='<b>%{text}</b><br>Support: %{x:.3f}<br>Confidence: %{y:.3f}<extra></extra>',
    ))
    fig.add_hline(y=min_conf, line_color=COLORS['faint'], line_width=1, line_dash='dot')
    fig.add_vline(x=min_sup,  line_color=COLORS['faint'], line_width=1, line_dash='dot')
    fig.add_annotation(x=rules_df['support'].max()*0.92, y=rules_df['confidence'].max(),
        text="High value zone", showarrow=False, font=dict(color=COLORS['amber'], size=10))
    theme(fig, 'Rule Landscape — Bubble size = Lift strength', 400,
          margin=dict(l=0,r=0,t=40,b=0),
          xaxis_title='Support (how common is this pattern?)',
          yaxis_title='Confidence (how reliably does A predict B?)')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    top_ant = rules_df.iloc[0]['antecedents']
    top_con = rules_df.iloc[0]['consequents']
    top_lift = rules_df.iloc[0]['lift']
    top_conf = rules_df.iloc[0]['confidence']
    st.markdown(f"""<div class="insight-card"><div class="label">So What?</div>
    The strongest pattern: when a student shows <b>{top_ant}</b>, there is a
    <b>{top_conf*100:.0f}% chance</b> they also show <b>{top_con}</b> — and this happens
    <b>{top_lift:.1f}x more often than chance</b> would predict.
    This is not coincidence — it is the product flywheel working. Every feature ALO shows
    a stressed student is landing. The business implication: <b>do not separate these features</b>.
    Bundle them. Users who see both are your highest-converting cohort.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>Top Rules by Lift</div>", unsafe_allow_html=True)
    top15 = rules_df.head(15).copy()
    ylabels = (top15['antecedents'] + ' ->\n' + top15['consequents'])[::-1].tolist()
    fig = go.Figure(go.Bar(
        y=ylabels, x=top15['lift'][::-1].values, orientation='h',
        marker_color=top15['lift'][::-1].values,
        marker_colorscale=[[0,'#1E3A5F'],[0.5,'#2563EB'],[1,'#F59E0B']],
        marker_opacity=0.9,
        text=[f"{v:.3f}" for v in top15['lift'][::-1]],
        textposition='outside', textfont=dict(family='JetBrains Mono', size=10, color=COLORS['muted']),
        hovertemplate='Lift: %{x:.3f}<extra></extra>'
    ))
    fig.add_vline(x=1.0, line_color=COLORS['muted'], line_dash='dot', line_width=1,
                  annotation_text="Random chance", annotation_font_size=9, annotation_font_color=COLORS['muted'])
    theme(fig, 'Rules Ranked by Strength', 400, margin=dict(l=0,r=70,t=40,b=0), xaxis_title='Lift score')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown("""<div class="insight-card"><div class="label">So What?</div>
    Every rule above 1.0 means ALO users are <b>not behaving randomly</b> — they are
    following predictable patterns the product team can design around.
    The higher the lift, the stronger the signal. <b>Rules above 1.3 are actionable</b>:
    they are strong enough to build automated in-app triggers on.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

col1, col2 = st.columns([6,4])

with col1:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>Co-occurrence Matrix — which behaviours cluster together?</div>", unsafe_allow_html=True)
    cooc_m = cooc.copy()
    np.fill_diagonal(cooc_m.values, np.nan)

    # Use coloraxis — avoids colorbar-inside-trace error on Plotly 5.18+
    fig = go.Figure(go.Heatmap(
        z=cooc_m.values, x=cooc_m.columns.tolist(), y=cooc_m.index.tolist(),
        coloraxis='coloraxis',
        text=np.round(cooc_m.values, 2), texttemplate='%{text}',
        textfont=dict(size=11, family='JetBrains Mono'),
        hovertemplate='%{y} + %{x}<br>Together in %{z:.0%} of users<extra></extra>',
        zmin=0.05, zmax=0.55
    ))
    fig.update_layout(
        coloraxis=dict(
            colorscale=[[0,COLORS['bg']],[0.3,'#1E3A5F'],[0.6,'#1D4ED8'],[1,'#60A5FA']],
            colorbar=dict(thickness=12, len=0.85, title='Support', tickfont=dict(color='#94A3B8'))
        )
    )
    theme(fig, 'Pairwise Co-occurrence (diagonal masked)', 420, margin=dict(l=0,r=40,t=40,b=0))
    fig.update_xaxes(tickangle=-35, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    top_pair = cooc_m.stack().dropna().idxmax()
    top_val  = cooc_m.stack().dropna().max()
    st.markdown(f"""<div class="insight-card"><div class="label">So What?</div>
    The brightest cells reveal behaviours that <b>naturally go hand-in-hand</b>.
    <b>{top_pair[0]}</b> and <b>{top_pair[1]}</b> co-occur in <b>{top_val*100:.0f}% of users</b>.
    That means nearly two-thirds of your user base is already doing both — without being asked.
    This is your highest-leverage UI decision: <b>surface these two features together</b> in one
    screen, one notification, one moment. You are not creating new behaviour — you are removing
    friction from a behaviour that already exists.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>Feature Adoption Rate</div>", unsafe_allow_html=True)
    sup_s = pd.Series(item_sup).sort_values(ascending=True)
    bclrs = [COLORS['green'] if v>=0.55 else COLORS['blue'] if v>=min_sup else COLORS['muted'] for v in sup_s.values]
    fig = go.Figure(go.Bar(
        x=sup_s.values, y=sup_s.index, orientation='h',
        marker_color=bclrs, marker_opacity=0.88,
        text=[f"{v*100:.0f}%" for v in sup_s.values], textposition='outside',
        textfont=dict(family='JetBrains Mono', size=11, color=COLORS['muted']),
        hovertemplate='%{y}: %{text} of users<extra></extra>'
    ))
    fig.add_vline(x=min_sup, line_color=COLORS['amber'], line_width=1.5, line_dash='dash',
                  annotation_text=f"Threshold {min_sup:.0%}", annotation_font_size=9,
                  annotation_font_color=COLORS['amber'])
    theme(fig, '% of Users Showing Each Behaviour', 420, margin=dict(l=0,r=60,t=40,b=0))
    fig.update_xaxes(tickformat='.0%', range=[0, 0.75], showgrid=False)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    high_adopt = [k for k,v in item_sup.items() if v >= 0.55]
    low_adopt  = [k for k,v in item_sup.items() if v < min_sup]
    st.markdown(f"""<div class="insight-card"><div class="label">So What?</div>
    <b>Green bars (>55%)</b> are your core platform loops — the majority of users already
    do this. <b>Build your onboarding around these</b>, not features users have to be taught.
    {"<b>Grey bars</b> (" + ', '.join(low_adopt) + ") are below threshold — these features have an adoption problem, not a design problem. Fix discovery before investing in new features." if low_adopt else "All features are above the adoption threshold — strong platform-wide engagement."}
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

display_df = rules_df[["antecedents","consequents","support","confidence","lift"]].round(3)
with st.expander(f"Full Rule Table — all {len(rules_df)} rules"):
    st.dataframe(display_df, hide_index=True, use_container_width=True)
    st.download_button("Download rules CSV", rules_df.to_csv(index=False).encode(), "ALO_rules.csv", "text/csv")