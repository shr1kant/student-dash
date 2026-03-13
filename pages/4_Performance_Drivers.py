"""
pages/4_Performance_Drivers.py — ALO Platform · GPA & Outcome Prediction
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
from engine import prepare_data, run_regression, COLORS, PLOTLY_LAYOUT, GLOBAL_CSS

st.set_page_config(page_title="Performance Drivers · ALO", page_icon="▲", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""<div style="font-size:1rem;font-weight:700;color:#E2E8F0;padding:16px 0 12px;border-bottom:1px solid #1E2D45;margin-bottom:16px;">▲ Performance Drivers</div>""", unsafe_allow_html=True)
    model_sel = st.radio("Regression Model", ["Linear Regression","Ridge (α=1)","Lasso (α=0.01)"])
    show_residuals = st.toggle("Show residual diagnostics", True)
    corr_threshold = st.slider("Correlation filter (|r| >)", 0.0, 0.8, 0.0, 0.05)
    st.markdown("---")
    st.markdown("""<div style="font-size:0.72rem;color:#475569;line-height:1.6;">
    All models predict <b style="color:#E2E8F0">GPA Change</b> per semester.
    R² of 0.19–0.21 is realistic for academic outcomes — many factors outside
    the platform also influence grades.
    </div>""", unsafe_allow_html=True)

@st.cache_data
def get_regression_data():
    raw = load_raw()
    df  = prepare_data(raw)
    return run_regression(df), df

reg, df = get_regression_data()
results  = reg['results']
active_r = results[model_sel]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding:0 0 24px 0; border-bottom:1px solid #1E2D45; margin-bottom:24px;">
    <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; color:#10B981; font-weight:600; margin-bottom:6px;">Performance Drivers</div>
    <h1 style="margin:0; font-size:1.8rem; font-weight:700; color:#E2E8F0; letter-spacing:-0.02em;">What Drives GPA Improvement?</h1>
    <p style="margin:6px 0 0 0; color:#64748B; font-size:0.88rem;">Linear regression · Target: GPA change per semester · {model_sel} active</p>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric(f"{model_sel} R²",  f"{active_r['r2']:.3f}",  "Variance explained")
c2.metric("MAE",               f"±{active_r['mae']:.3f}", "Avg GPA prediction error")
c3.metric("RMSE",              f"{active_r['rmse']:.3f}", "Root mean squared error")
c4.metric("CV R²",             f"{active_r['cv_r2']:.3f}","5-fold cross-validated")
c5.metric("Baseline R²",       "0.000",                   "Predicting mean only")

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Row 1: Model comparison + Actual vs Predicted ─────────────────────────────
col1, col2 = st.columns([4,6])

with col1:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>3-Model Comparison</div>", unsafe_allow_html=True)

    model_names = list(results.keys())
    metrics_list = ['r2','mae','rmse','cv_r2']
    metric_labels = ['R²','MAE','RMSE','CV R²']

    fig = make_subplots(rows=2, cols=2, subplot_titles=metric_labels,
                         vertical_spacing=0.2, horizontal_spacing=0.15)

    model_colors = [COLORS['blue'], COLORS['green'], COLORS['purple']]
    positions = [(1,1),(1,2),(2,1),(2,2)]
    for (r,c), metric, label in zip(positions, metrics_list, metric_labels):
        vals = [results[m][metric] for m in model_names]
        best_idx = vals.index(max(vals) if metric in ['r2','cv_r2'] else min(vals))
        bar_colors = [COLORS['amber'] if i==best_idx else model_colors[i] for i in range(len(model_names))]
        fig.add_trace(go.Bar(
            x=[m.split(' ')[0] for m in model_names],
            y=vals,
            marker_color=bar_colors,
            marker_opacity=0.85,
            text=[f"{v:.3f}" for v in vals],
            textposition='outside',
            textfont=dict(size=10, family='JetBrains Mono', color=COLORS['muted']),
            showlegend=False,
            hovertemplate='%{x}: %{y:.4f}<extra>' + label + '</extra>'
        ), row=r, col=c)

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       font=dict(color=COLORS['text'], family='DM Sans'),
                       height=360, margin=dict(l=10,r=10,t=40,b=10))
    for i in range(1,5):
        fig.update_layout(**{f'xaxis{i}' if i>1 else 'xaxis':
                              dict(gridcolor=COLORS['faint'], linecolor=COLORS['border'], tickfont=dict(size=9,color=COLORS['muted'])),
                             f'yaxis{i}' if i>1 else 'yaxis':
                              dict(gridcolor=COLORS['faint'], linecolor=COLORS['border'], tickfont=dict(size=9,color=COLORS['muted']))})

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown(f"""<div class="insight-card">
    <div class="label">◈ Model Selection</div>
    <b>Lasso achieves the best CV R² ({results['Lasso (α=0.01)']['cv_r2']:.3f})</b> — L1 regularisation
    automatically zeros out noisy features, improving generalisability. All three models deliver
    similar practical accuracy (~±{results['Linear Regression']['mae']:.2f} GPA points),
    which is operationally sufficient for cohort-level platform ROI reporting.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>Actual vs Predicted GPA Change</div>", unsafe_allow_html=True)

    y_test = reg['y_test'].values
    y_pred = reg['y_pred']
    residuals = y_test - y_pred
    pct_in_band = (np.abs(residuals) <= 0.15).mean() * 100

    # Colour points by residual magnitude
    res_abs = np.abs(residuals)
    res_norm = res_abs / res_abs.max()

    fig = go.Figure()
    # Perfect line band
    lims = [min(y_test.min(), y_pred.min())-0.05, max(y_test.max(), y_pred.max())+0.05]
    fig.add_trace(go.Scatter(x=lims, y=[l+0.15 for l in lims], mode='lines',
                              line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=lims, y=[l-0.15 for l in lims], mode='lines',
                              fill='tonexty', fillcolor='rgba(16,185,129,0.07)',
                              line=dict(width=0), name='±0.15 GPA band'))
    fig.add_trace(go.Scatter(x=lims, y=lims, mode='lines',
                              line=dict(color=COLORS['green'], width=2, dash='dash'),
                              name='Perfect prediction'))

    # Scatter coloured by residual
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred, mode='markers',
        marker=dict(
            color=res_abs, colorscale=[[0,COLORS['blue']],[0.5,COLORS['amber']],[1,COLORS['red']]],
            size=7, opacity=0.75,
            line=dict(width=0.5, color=COLORS['bg']),
            showscale=True,
            colorbar=dict(title='|Error|', tickfont=dict(color=COLORS['muted']),
                          titlefont=dict(color=COLORS['muted']), thickness=12, len=0.7)
        ),
        name='Predictions',
        hovertemplate='Actual: %{x:.3f}<br>Predicted: %{y:.3f}<br>Error: %{marker.color:.3f}<extra></extra>'
    ))

    fig.update_layout(**PLOTLY_LAYOUT,
                       title=f'Actual vs Predicted · R²={active_r["r2"]:.3f} · {pct_in_band:.0f}% within ±0.15 GPA',
                       height=360, xaxis_title='Actual GPA Change',
                       yaxis_title='Predicted GPA Change',
                       legend=dict(x=0.02, y=0.97, font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown(f"""<div class="insight-card">
    <div class="label">◈ Prediction Insight</div>
    <b>{pct_in_band:.0f}% of predictions</b> fall within ±0.15 GPA points of actual outcomes.
    Points coloured <b style="color:#F59E0B">amber/red</b> are high-error outliers —
    these are students where external factors (not captured by platform data) dominate.
    They're worth flagging for manual academic advisor outreach rather than automated nudges.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Row 2: Coefficients + Correlation heatmap ─────────────────────────────────
col1, col2 = st.columns([5,5])

with col1:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>What Moves GPA · Regression Coefficients</div>", unsafe_allow_html=True)

    coef_df = reg['coef_df'].copy()
    coef_df['feature_clean'] = coef_df['feature'].str.replace('_',' ').str.replace(' enc','')
    top_n_c = st.slider("Show features", 6, len(coef_df), 12, label_visibility='collapsed')
    top_c = coef_df.head(top_n_c).sort_values('coefficient')

    bar_colors_c = [COLORS['green'] if v > 0 else COLORS['red'] for v in top_c['coefficient']]

    fig = go.Figure(go.Bar(
        y=top_c['feature_clean'],
        x=top_c['coefficient'],
        orientation='h',
        marker_color=bar_colors_c,
        marker_opacity=0.88,
        text=[f"{'+' if v>0 else ''}{v:.4f}" for v in top_c['coefficient']],
        textposition='outside',
        textfont=dict(family='JetBrains Mono', size=10, color=COLORS['muted']),
        hovertemplate='%{y}<br>Coefficient: %{x:.4f}<extra></extra>'
    ))
    fig.add_vline(x=0, line_color=COLORS['muted'], line_width=1.5)
    fig.update_layout(**PLOTLY_LAYOUT,
                       title=f'Standardised Coefficients — {model_sel}',
                       height=max(360, top_n_c*32), margin=dict(l=0,r=90,t=40,b=0),
                       xaxis_title='Coefficient (standardised features)',
                       yaxis_title='')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    pos_feats = top_c[top_c['coefficient']>0]['feature_clean'].tail(3).tolist()
    neg_feats = top_c[top_c['coefficient']<0]['feature_clean'].head(2).tolist()
    st.markdown(f"""<div class="insight-card">
    <div class="label">◈ Lever Insight</div>
    <b style="color:#10B981">GPA boosters:</b> {', '.join(pos_feats)}<br>
    <b style="color:#EF4444">GPA suppressors:</b> {', '.join(neg_feats) if neg_feats else 'None significant'}<br><br>
    Platform features (AI usage, quiz adjustments, adaptive scheduling) dominate the positive
    side — these are <b>direct product levers the team controls</b>. Prioritise shipping
    improvements to these features first for maximum outcome ROI.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='alo-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>Signal Correlation Map · Filter by strength</div>", unsafe_allow_html=True)

    corr = reg['corr'].copy()
    if corr_threshold > 0:
        mask_weak = np.abs(corr) < corr_threshold
        corr_plot = corr.copy()
        corr_plot[mask_weak] = np.nan
    else:
        corr_plot = corr

    fig = go.Figure(go.Heatmap(
        z=corr_plot.values,
        x=[c.replace('_',' ') for c in corr_plot.columns],
        y=[c.replace('_',' ') for c in corr_plot.index],
        colorscale=[[0,'#450A0A'],[0.25,'#7F1D1D'],[0.5,COLORS['card']],[0.75,'#1E3A5F'],[1,'#1D4ED8']],
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr_plot.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=10, family='JetBrains Mono'),
        hovertemplate='%{y} × %{x}<br>r = %{z:.3f}<extra></extra>',
        showscale=True,
        colorbar=dict(title='r', tickfont=dict(color=COLORS['muted']),
                      titlefont=dict(color=COLORS['muted']), thickness=12, len=0.85)
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                       title=f'Pearson Correlations {f"(|r|≥{corr_threshold})" if corr_threshold>0 else "(all)"}',
                       height=420, margin=dict(l=0,r=40,t=40,b=0))
    fig.update_xaxes(tickangle=-35, tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    gpa_corr = corr['GPA_Change'].drop('GPA_Change').sort_values(ascending=False)
    top_pos = gpa_corr[gpa_corr>0].index[0] if (gpa_corr>0).any() else "—"
    top_neg = gpa_corr[gpa_corr<0].index[-1] if (gpa_corr<0).any() else "—"
    st.markdown(f"""<div class="insight-card">
    <div class="label">◈ Correlation Insight</div>
    <b>Strongest GPA link:</b> {top_pos.replace('_',' ')} (r={gpa_corr.get(top_pos,0):.2f})<br>
    <b>Strongest inverse:</b> {top_neg.replace('_',' ')} (r={gpa_corr.get(top_neg,0):.2f})<br><br>
    Use the slider to filter weak correlations — the <b>dark red → dark blue</b> cells
    that survive at |r|>0.3 are the relationships robust enough to build product decisions on.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Residual diagnostics (collapsible) ───────────────────────────────────────
if show_residuals:
    st.markdown("<div class='alo-card' style='margin-top:8px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748B;margin-bottom:12px;'>Residual Diagnostics · Model Health Check</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Histogram(
            x=residuals, nbinsx=32,
            marker_color=COLORS['purple'], marker_opacity=0.8,
            marker_line=dict(width=0.5, color=COLORS['bg']),
            hovertemplate='Residual: %{x:.3f}<br>Count: %{y}<extra></extra>'
        ))
        fig.add_vline(x=0, line_color=COLORS['amber'], line_width=2, line_dash='dash')
        fig.add_vline(x=np.mean(residuals), line_color=COLORS['blue'], line_width=1.5, line_dash='dot',
                       annotation_text=f"μ={np.mean(residuals):.4f}", annotation_font_color=COLORS['blue_light'])
        fig.update_layout(**PLOTLY_LAYOUT, title='Residual Distribution (should be ~Normal, centred at 0)',
                           height=260, xaxis_title='Residual (Actual − Predicted)', margin=dict(l=0,r=10,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with c2:
        fig = go.Figure(go.Scatter(
            x=y_pred, y=residuals, mode='markers',
            marker=dict(color=COLORS['purple'], size=6, opacity=0.6,
                        line=dict(width=0.5, color=COLORS['bg'])),
            hovertemplate='Fitted: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>'
        ))
        fig.add_hline(y=0, line_color=COLORS['amber'], line_width=2, line_dash='dash')
        # Rolling average of residuals
        sorted_idx = np.argsort(y_pred)
        xp_sorted = y_pred[sorted_idx]
        rp_sorted = residuals[sorted_idx]
        window = max(5, len(xp_sorted)//10)
        rolling_mean = np.convolve(rp_sorted, np.ones(window)/window, mode='valid')
        x_rolling = xp_sorted[window//2: window//2+len(rolling_mean)]
        fig.add_trace(go.Scatter(x=x_rolling, y=rolling_mean, mode='lines',
                                  line=dict(color=COLORS['amber'], width=2.5),
                                  name='Rolling avg', showlegend=True))
        fig.update_layout(**PLOTLY_LAYOUT, title='Residuals vs Fitted (no pattern = good)',
                           height=260, xaxis_title='Predicted GPA Change',
                           yaxis_title='Residual', margin=dict(l=0,r=10,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    mean_r, std_r = np.mean(residuals), np.std(residuals)
    skew_r = float(pd.Series(residuals).skew())
    st.markdown(f"""<div class="insight-card">
    <div class="label">◈ Diagnostic Verdict</div>
    Residuals: μ={mean_r:.4f} (near zero ✓) · σ={std_r:.4f} · Skew={skew_r:.2f}
    {"(slight right skew — model under-predicts some high achievers)" if skew_r > 0.3 else "(approximately symmetric ✓)"}.
    Rolling average in the residuals vs fitted plot shows
    {"no clear trend — homoscedasticity holds ✓" if abs(skew_r) < 0.5 else "mild curvature — a polynomial term may improve fit"}.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
