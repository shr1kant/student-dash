"""
pages/3_Behaviour_Patterns.py — ALO Platform · Behavioural Pattern Mining
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_embedded import load_raw
from engine import prepare_data, run_arm, COLORS, PLOTLY_LAYOUT, GLOBAL_CSS

st.set_page_config(page_title="Behaviour Patterns · ALO", page_icon="◆", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""<div style="font-size:1rem;font-weight:700;color:#E2E8F0;padding:16px 0 12px;border-bottom:1px solid #1E2D45;margin-bottom:16px;">◆ Behaviour Patterns</div>""", unsafe_allow_html=True)
    min_sup  = st.slider("Min Support",    0.20, 0.60, 0.28, 0.02)
    min_conf = st.slider("Min Confidence", 0.50, 0.95, 0.60, 0.05)
    min_lift = st.slider("Min Lift",       1.00, 2.50, 1.10, 0.05)
    st.markdown("---")
    st.markdown("""<div style="font-size:0.72rem;color:#475569;line-height:1.6;">
    <b style="color:#94A3B8">Support</b> = % of users showing both behaviours<br>
    <b style="color:#94A3B8">Confidence</b> = P(B|A) — how often B follows A<br>
    <b style="color:#94A3B8">Lift > 1</b> = co-occurrence beyond chance<br><br>
    Adjust sliders to explore different behavioural thresholds.
    </div>""", unsafe_allow_html=True)

@st.cache_data
def get_patterns(sup, conf, lift):
    raw = load_raw()
    df  = prepare_data(raw)
    return run_arm(df, min_sup=sup, min_conf=conf, min_lift=lift), df

(rules_df, item_sup, cooc), df = get_patterns(min_sup, min_conf, min_lift)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding:0 0 24px 0; border-bottom:1px solid #1E2D45; margin-bottom:24px;">
    <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; color:#14B8A6; font-weight:600; margin-bottom:6px;">Behaviour Patterns</div>
    <h1 style="margin:0; font-size:1.8rem; font-weight:700; color:#E2E8F0; letter-spacing:-0.02em;">What Do Users Do Together?</h1>
    <p style="margin:6px 0 0 0; color:#64748B; font-size:0.88rem;">Apriori association mining · {len(rules_df)} rules found · Support≥{min_sup} · Confidence≥{min_conf} · Lift≥{min_lift}</p>
</div>
""", unsafe_allow_html=True)

if len(rules_df) == 0:
    st.warning("No rules found at current thresholds. Try lowering support or confidence in the sidebar.")
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
c1.metric("Rules Found",     str(len(rules_df)))
c2.metric("Top Lift",        f"{rules_df['lift'].max():.3f}")
c3.metric("Avg Confidence",  f"{rules_df['confidence'].mean():.2f}")
c4.metric("Strongest Pattern", f"{rules_df.iloc[0]['antecedents']} → {rules_df.iloc[0]['consequents']}"[:38]+"…")

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Row 1: Bubble chart (interactive) + top rules ─────────────────────────────
col1, col2 = st.columns([6,4])

with col1:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>Rule Map · Support × Confidence × Lift — hover & click</div>", unsafe_allow_html=True)

    rules_df['Rule'] = rules_df['antecedents'] + ' → ' + rules_df['consequents']
    rules_df['Lift_norm'] = (rules_df['lift'] - rules_df['lift'].min()) / (rules_df['lift'].max() - rules_df['lift'].min() + 1e-9)

    fig = go.Figure(go.Scatter(
        x=rules_df['support'],
        y=rules_df['confidence'],
        mode='markers',
        marker=dict(
            size=rules_df['lift'] * 20,
            color=rules_df['lift'],
            colorscale=[[0,'#1E3A5F'],[0.5,'#2563EB'],[1,'#F59E0B']],
            showscale=True,
            colorbar=dict(title='Lift', tickfont=dict(color=COLORS['muted']),
                          titlefont=dict(color=COLORS['muted']), thickness=12, len=0.8),
            opacity=0.85,
            line=dict(width=1, color=COLORS['bg'])
        ),
        text=rules_df['Rule'],
        hovertemplate=(
            '<b>%{text}</b><br>'
            'Support: %{x:.3f}<br>'
            'Confidence: %{y:.3f}<br>'
            'Lift: %{marker.color:.3f}<extra></extra>'
        ),
        customdata=rules_df[['antecedents','consequents','lift']].values
    ))

    # Quadrant lines
    fig.add_hline(y=min_conf, line_color=COLORS['faint'], line_width=1, line_dash='dot')
    fig.add_vline(x=min_sup,  line_color=COLORS['faint'], line_width=1, line_dash='dot')
    fig.add_annotation(x=rules_df['support'].max()*0.95, y=rules_df['confidence'].max(),
                        text="High value zone", showarrow=False,
                        font=dict(color=COLORS['amber'], size=10))

    fig.update_layout(**PLOTLY_LAYOUT,
                       title='Rule Landscape · Size & Color = Lift strength',
                       height=400, xaxis_title='Support (population coverage)',
                       yaxis_title='Confidence (predictive power)')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("""<div class="insight-card">
    <div class="label">◈ Product Loop Insight</div>
    Rules in the <b>top-right zone</b> (high support + high confidence) are the platform's
    most reliable behavioural loops — they occur frequently AND predict strongly.
    <b>Bubble size indicates lift</b>: larger bubbles reveal rare but powerful co-occurrences
    worth designing features around (e.g. targeted stress-to-AI nudges).
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>Top Rules by Lift</div>", unsafe_allow_html=True)

    top15 = rules_df.head(15).copy()

    fig = go.Figure(go.Bar(
        y=(top15['antecedents'] + ' →<br>' + top15['consequents'])[::-1],
        x=top15['lift'][::-1],
        orientation='h',
        marker=dict(
            color=top15['lift'][::-1],
            colorscale=[[0,'#1E3A5F'],[0.5,'#2563EB'],[1,'#F59E0B']],
            showscale=False,
            opacity=0.9,
            line=dict(width=0)
        ),
        text=[f"{v:.3f}" for v in top15['lift'][::-1]],
        textposition='outside',
        textfont=dict(family='JetBrains Mono', size=10, color=COLORS['muted']),
        hovertemplate='Lift: %{x:.3f}<br>%{y}<extra></extra>'
    ))
    fig.add_vline(x=1.0, line_color=COLORS['muted'], line_dash='dot', line_width=1,
                   annotation_text="Baseline", annotation_font_size=9, annotation_font_color=COLORS['muted'])
    fig.update_layout(**PLOTLY_LAYOUT,
                       title='Lift Score · Rules Ranked',
                       height=400, margin=dict(l=0,r=70,t=40,b=0),
                       xaxis_title='Lift')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

# ── Row 2: Co-occurrence heatmap + item frequency ─────────────────────────────
col1, col2 = st.columns([6,4])

with col1:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>Co-occurrence Matrix · How behaviours cluster together</div>", unsafe_allow_html=True)

    cooc_mask = cooc.copy()
    np.fill_diagonal(cooc_mask.values, np.nan)

    fig = go.Figure(go.Heatmap(
        z=cooc_mask.values,
        x=cooc_mask.columns.tolist(),
        y=cooc_mask.index.tolist(),
        colorscale=[[0, COLORS['bg']], [0.3,'#1E3A5F'], [0.6,'#1D4ED8'], [1,'#60A5FA']],
        text=np.round(cooc_mask.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=11, family='JetBrains Mono'),
        hovertemplate='%{y} × %{x}<br>Co-occurrence: %{z:.2f}<extra></extra>',
        showscale=True,
        colorbar=dict(tickfont=dict(color=COLORS['muted']), thickness=12, len=0.85,
                      title='Support', titlefont=dict(color=COLORS['muted'])),
        zmin=0.05, zmax=0.55,
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                       title='Pairwise Co-occurrence (diagonal masked)',
                       height=420, margin=dict(l=0,r=40,t=40,b=0))
    fig.update_xaxes(tickangle=-35, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    top_pair = cooc_mask.stack().dropna().idxmax()
    top_val  = cooc_mask.stack().dropna().max()
    st.markdown(f"""<div class="insight-card">
    <div class="label">◈ Feature Bundling Insight</div>
    <b>{top_pair[0]}</b> and <b>{top_pair[1]}</b> show the strongest co-occurrence
    ({top_val:.2f} — occurs together in {top_val*100:.0f}% of all user sessions).
    This is ALO's most reliable product pairing: surfacing these behaviours together
    in the UI (e.g. triggering AI resources when stress signals spike) will reinforce
    the loop that's already working organically.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>Feature Adoption Rate</div>", unsafe_allow_html=True)

    sup_s = pd.Series(item_sup).sort_values(ascending=True)
    threshold = min_sup
    bar_colors = [COLORS['green'] if v >= 0.55 else COLORS['blue'] if v >= threshold else COLORS['muted']
                  for v in sup_s.values]

    fig = go.Figure(go.Bar(
        x=sup_s.values,
        y=sup_s.index,
        orientation='h',
        marker_color=bar_colors,
        marker_opacity=0.88,
        text=[f"{v*100:.0f}%" for v in sup_s.values],
        textposition='outside',
        textfont=dict(family='JetBrains Mono', size=11, color=COLORS['muted']),
        hovertemplate='%{y}: %{x:.2f} (%{text} of users)<extra></extra>'
    ))
    fig.add_vline(x=threshold, line_color=COLORS['amber'], line_width=1.5, line_dash='dash',
                   annotation_text=f"Min support {threshold}", annotation_font_size=9,
                   annotation_font_color=COLORS['amber'])
    fig.update_layout(**PLOTLY_LAYOUT,
                       title='% of Users Showing Each Behaviour',
                       height=420, margin=dict(l=0,r=60,t=40,b=0),
                       xaxis=dict(tickformat='.0%', range=[0,0.75], showgrid=False))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    high_adopt = [k for k,v in item_sup.items() if v >= 0.55]
    low_adopt  = [k for k,v in item_sup.items() if v < threshold]
    st.markdown(f"""<div class="insight-card">
    <div class="label">◈ Adoption Insight</div>
    <b>High adoption (>55%):</b> {', '.join(high_adopt) if high_adopt else 'None at this threshold'}<br>
    <b>Below threshold:</b> {', '.join(low_adopt) if low_adopt else 'None'}<br><br>
    Design your default onboarding flow around high-adoption behaviours.
    Low-adoption features need activation nudges, not feature cuts.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── All rules table ───────────────────────────────────────────────────────────
with st.expander(f"📋 All {len(rules_df)} Rules — full table"):
    st.dataframe(
        rules_df[['antecedents','consequents','support','confidence','lift']]
            .style.background_gradient(subset=['lift'],       cmap='YlOrRd')
                   .background_gradient(subset=['confidence'],cmap='Blues')
                   .format({'support':'{:.3f}','confidence':'{:.3f}','lift':'{:.3f}'}),
        hide_index=True, use_container_width=True
    )
    csv = rules_df.to_csv(index=False).encode()
    st.download_button("⬇ Export rules CSV", csv, "ALO_rules.csv", "text/csv")
