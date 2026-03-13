"""
engine.py — ALO Platform · Shared Analytics & Style Engine
All ML computation + global CSS/Plotly theming lives here.
"""

import pandas as pd
import numpy as np
import warnings
from itertools import combinations
try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    go = None
    px = None
warnings.filterwarnings('ignore')

# ── Design System ─────────────────────────────────────────────────────────────
COLORS = {
    'bg':        '#0A0F1E',
    'card':      '#111827',
    'card2':     '#1A2235',
    'border':    '#1E2D45',
    'blue':      '#3B82F6',
    'blue_light':'#60A5FA',
    'amber':     '#F59E0B',
    'green':     '#10B981',
    'red':       '#EF4444',
    'purple':    '#8B5CF6',
    'teal':      '#14B8A6',
    'text':      '#E2E8F0',
    'muted':     '#94A3B8',
    'faint':     '#1E293B',
}

PERSONA_PALETTE = {
    'Efficient Achievers':   '#10B981',
    'High-Stress Engagers':  '#EF4444',
    'Passive Learners':      '#94A3B8',
    'Developing Strivers':   '#3B82F6',
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans, sans-serif', color=COLORS['text'], size=12),
    margin=dict(l=10, r=10, t=48, b=10),
    legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0),
    xaxis=dict(gridcolor=COLORS['faint'], linecolor=COLORS['border'], tickfont=dict(color=COLORS['muted'])),
    yaxis=dict(gridcolor=COLORS['faint'], linecolor=COLORS['border'], tickfont=dict(color=COLORS['muted'])),
    title_font=dict(size=14, color=COLORS['text']),
)

def apply_theme(fig, title='', height=400):
    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=height)
    return fig

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 1400px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #080D1A !important;
    border-right: 1px solid #1E2D45;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span,
[data-testid="stSidebar"] label { color: #94A3B8 !important; font-size: 0.82rem; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #E2E8F0 !important; }

/* Nav pills in sidebar */
[data-testid="stSidebarNav"] a {
    border-radius: 8px;
    margin: 2px 0;
    padding: 6px 12px;
    transition: all 0.15s;
}
[data-testid="stSidebarNav"] a:hover { background: #1E293B !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1E2D45;
    border-radius: 12px;
    padding: 16px !important;
}
[data-testid="metric-container"] label { color: #94A3B8 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; color: #E2E8F0 !important; font-size: 1.7rem !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* Tabs */
[data-testid="stTabs"] button {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #64748B !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 16px !important;
    transition: all 0.15s;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #3B82F6 !important;
    border-bottom: 2px solid #3B82F6 !important;
    background: #111827 !important;
}

/* Sliders */
[data-testid="stSlider"] > div > div > div > div { background: #3B82F6 !important; }

/* Select boxes */
[data-testid="stSelectbox"] > div { background: #111827 !important; border-color: #1E2D45 !important; }

/* Dataframes */
[data-testid="stDataFrame"] { border: 1px solid #1E2D45; border-radius: 10px; overflow: hidden; }

/* Expanders */
details { background: #111827 !important; border: 1px solid #1E2D45 !important; border-radius: 10px !important; }

/* Custom component styles */
.alo-card {
    background: #111827;
    border: 1px solid #1E2D45;
    border-radius: 14px;
    padding: 20px 24px;
    margin: 6px 0;
}
.alo-page-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 20px 0 24px 0;
    border-bottom: 1px solid #1E2D45;
    margin-bottom: 24px;
}
.alo-page-header .icon {
    font-size: 2rem;
    width: 52px;
    height: 52px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
    background: #1E293B;
}
.alo-page-header h1 { margin: 0; font-size: 1.5rem; font-weight: 700; color: #E2E8F0; }
.alo-page-header p  { margin: 2px 0 0 0; font-size: 0.85rem; color: #64748B; }

.insight-card {
    background: linear-gradient(135deg, #0F2044 0%, #111827 100%);
    border: 1px solid #1E4080;
    border-left: 3px solid #3B82F6;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.85rem;
    line-height: 1.6;
    color: #CBD5E1;
}
.insight-card .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #3B82F6;
    font-weight: 600;
    margin-bottom: 4px;
}
.insight-card b { color: #E2E8F0; }

.stat-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    margin: 0 2px;
}
.pill-blue   { background: #1E3A5F; color: #60A5FA; }
.pill-green  { background: #064E3B; color: #34D399; }
.pill-amber  { background: #451A03; color: #FBBF24; }
.pill-red    { background: #450A0A; color: #F87171; }

.persona-badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    display: inline-block;
}
</style>
"""

# ── Data Preparation ──────────────────────────────────────────────────────────

def prepare_data(df_raw):
    df = df_raw.copy()
    df.drop_duplicates(inplace=True)
    df['Major'] = df['Major'].str.strip().str.title()

    df['Study_Hours_Per_Week'] = df.groupby('Major')['Study_Hours_Per_Week'].transform(
        lambda x: x.fillna(x.median()))
    df['Assignment_Completion_Rate'] = df.groupby('Year_of_Study')['Assignment_Completion_Rate'].transform(
        lambda x: x.fillna(x.median()))
    df['_sb'] = pd.cut(df['Stress_Level'], bins=[0,3,6,10], labels=['Low','Mid','High'])
    for col in ['Cognitive_Load_Score','AI_Usage_Hours_Per_Week']:
        df[col] = df.groupby('_sb')[col].transform(lambda x: x.fillna(x.median()))
    df.drop('_sb', axis=1, inplace=True)
    df['_pq'] = pd.qcut(df['Productivity_Improvement'].rank(method='first'), q=4, labels=['Q1','Q2','Q3','Q4'])
    df['Satisfaction_Score'] = df.groupby('_pq')['Satisfaction_Score'].transform(
        lambda x: x.fillna(x.median()))
    df.drop('_pq', axis=1, inplace=True)
    df['Willingness_to_Pay'] = df['Willingness_to_Pay'].fillna(df['Willingness_to_Pay'].median())

    for col in ['Study_Hours_Per_Week','AI_Usage_Hours_Per_Week','Willingness_to_Pay']:
        df[col] = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))

    def mm(s): return (s - s.min()) / (s.max() - s.min())
    n = {c: mm(df[c]) for c in ['Study_Hours_Per_Week','AI_Usage_Hours_Per_Week',
                                   'Productivity_Improvement','Retention_Score']}
    sei = ((df['Average_Test_Score']/100) * df['Assignment_Completion_Rate']) / (n['Study_Hours_Per_Week'] + 0.01)
    df['Study_Efficiency_Index'] = (mm(sei.clip(upper=sei.quantile(0.99))) * 10).round(3)
    eng = (n['AI_Usage_Hours_Per_Week']
           + df['Adaptive_Schedule_Adjustments']/df['Adaptive_Schedule_Adjustments'].max()
           + df['Quiz_Frequency_Adjustment']/df['Quiz_Frequency_Adjustment'].max()
           + df['Resource_Recommendations']/df['Resource_Recommendations'].max()) / 4
    df['Engagement_Level'] = (eng * 10).round(3)
    df['Learning_Improvement_Index'] = (
        mm(df['GPA_Change'])*0.40 + n['Productivity_Improvement']*0.35 + n['Retention_Score']*0.25
    ) * 10
    df['Learning_Improvement_Index'] = df['Learning_Improvement_Index'].round(3)
    return df

# ── Clustering ────────────────────────────────────────────────────────────────

def run_clustering(df, k=4):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    feats = ['Study_Hours_Per_Week','Average_Test_Score','Stress_Level',
             'AI_Usage_Hours_Per_Week','Engagement_Level','Study_Efficiency_Index',
             'Learning_Improvement_Index','GPA_Change','Satisfaction_Score']
    X = df[feats].copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    inertias, silhouettes = [], []
    for ki in range(2, 9):
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        labs = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(Xs, labs))

    km_final = KMeans(n_clusters=k, random_state=42, n_init=10)
    df = df.copy()
    df['Cluster'] = km_final.fit_predict(Xs)

    centroids = pd.DataFrame(scaler.inverse_transform(km_final.cluster_centers_), columns=feats)
    eff_r = centroids['Study_Efficiency_Index'].rank()
    str_r = centroids['Stress_Level'].rank()
    gpa_r = centroids['GPA_Change'].rank()
    eng_r = centroids['Engagement_Level'].rank()

    persona_map = {}
    for i in range(k):
        if eff_r[i] >= 3 and gpa_r[i] >= 3:     persona_map[i] = 'Efficient Achievers'
        elif str_r[i] >= 3 and eng_r[i] >= 3:    persona_map[i] = 'High-Stress Engagers'
        elif eng_r[i] <= 2 and gpa_r[i] <= 2:    persona_map[i] = 'Passive Learners'
        else:                                      persona_map[i] = 'Developing Strivers'

    df['Persona'] = df['Cluster'].map(persona_map)

    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(Xs)
    df['PCA1'], df['PCA2'] = Xp[:,0], Xp[:,1]

    return df, dict(inertias=inertias, silhouettes=silhouettes, k_range=list(range(2,9)),
                    best_silhouette=silhouettes[k-2], pca_var=pca.explained_variance_ratio_.tolist(),
                    features=feats, centroids=centroids)

# ── Classification ────────────────────────────────────────────────────────────

def run_classification(df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  roc_auc_score, roc_curve, accuracy_score)
    df = df.copy()
    wtp_med = df['Willingness_to_Pay'].median()
    df['WTP_High'] = (df['Willingness_to_Pay'] >= wtp_med).astype(int)
    le = LabelEncoder()
    df['Major_enc'] = le.fit_transform(df['Major'])
    feats = ['Age','Year_of_Study','Study_Hours_Per_Week','Assignment_Completion_Rate',
             'Average_Test_Score','Retention_Score','Learning_Speed_Index','Cognitive_Load_Score',
             'Stress_Level','Adaptive_Schedule_Adjustments','Quiz_Frequency_Adjustment',
             'Resource_Recommendations','AI_Usage_Hours_Per_Week','Productivity_Improvement',
             'Satisfaction_Score','GPA_Change','Retention_Likelihood','Study_Efficiency_Index',
             'Engagement_Level','Learning_Improvement_Index','Major_enc']
    X, y = df[feats], df['WTP_High']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                 random_state=42, class_weight='balanced')
    rf.fit(Xtr_s, ytr)
    yp_rf, ypr_rf = rf.predict(Xte_s), rf.predict_proba(Xte_s)[:,1]

    lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr.fit(Xtr_s, ytr)
    yp_lr, ypr_lr = lr.predict(Xte_s), lr.predict_proba(Xte_s)[:,1]

    fpr_rf, tpr_rf, _ = roc_curve(yte, ypr_rf)
    fpr_lr, tpr_lr, _ = roc_curve(yte, ypr_lr)
    imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)

    return dict(wtp_med=wtp_med,
                rf=dict(acc=accuracy_score(yte,yp_rf), auc=roc_auc_score(yte,ypr_rf),
                        cm=confusion_matrix(yte,yp_rf), fpr=fpr_rf, tpr=tpr_rf,
                        report=classification_report(yte,yp_rf,target_names=['Low WTP','High WTP'],output_dict=True)),
                lr=dict(acc=accuracy_score(yte,yp_lr), auc=roc_auc_score(yte,ypr_lr),
                        cm=confusion_matrix(yte,yp_lr), fpr=fpr_lr, tpr=tpr_lr),
                importances=imp, df=df, y_test=yte)

# ── Association Rules ─────────────────────────────────────────────────────────

def run_arm(df, min_sup=0.28, min_conf=0.60, min_lift=1.10):
    bmap = {'AI_Usage_Hours_Per_Week':'AI Usage','Engagement_Level':'Engagement',
            'Stress_Level':'Stress','Average_Test_Score':'Test Score',
            'Productivity_Improvement':'Productivity','GPA_Change':'GPA Change',
            'Satisfaction_Score':'Satisfaction','Study_Efficiency_Index':'Study Efficiency',
            'Adaptive_Schedule_Adjustments':'Adaptive Adj','Willingness_to_Pay':'WTP'}
    df = df.copy()
    for col, label in bmap.items():
        df[f'{label}↑'] = (df[col] >= df[col].median()).astype(bool)
    icols = [f'{l}↑' for l in bmap.values()]
    basket = df[icols].astype(bool)
    n = len(basket)

    freq = {}
    for item in icols:
        s = basket[item].sum()/n
        if s >= min_sup: freq[frozenset([item])] = s
    for a,b in combinations(list(freq.keys()), 2):
        c = a|b
        s = basket[list(c)].all(axis=1).sum()/n
        if s >= min_sup: freq[c] = s
    seen3 = set()
    for a,b in combinations([k for k in freq if len(k)==2], 2):
        c=a|b
        if len(c)==3 and c not in seen3:
            seen3.add(c)
            s = basket[list(c)].all(axis=1).sum()/n
            if s >= min_sup: freq[c] = s

    rules = []
    for itemset, sxy in freq.items():
        if len(itemset)<2: continue
        for sz in range(1,len(itemset)):
            for ant in combinations(list(itemset),sz):
                ant=frozenset(ant); con=itemset-ant
                sx=freq.get(ant, basket[list(ant)].all(axis=1).sum()/n)
                if sx==0: continue
                conf=sxy/sx
                sy=freq.get(con, basket[list(con)].all(axis=1).sum()/n)
                lift=conf/sy if sy>0 else 0
                if conf>=min_conf and lift>=min_lift:
                    rules.append(dict(
                        antecedents=', '.join(sorted([i.replace('↑','') for i in ant])),
                        consequents=', '.join(sorted([i.replace('↑','') for i in con])),
                        support=round(sxy,4), confidence=round(conf,4), lift=round(lift,4)
                    ))

    rdf = pd.DataFrame(rules).sort_values('lift',ascending=False).reset_index(drop=True) if rules else pd.DataFrame(columns=['antecedents','consequents','support','confidence','lift'])
    item_sup = {c.replace('↑',''):basket[c].sum()/n for c in icols}
    cooc = basket.astype(int).T.dot(basket.astype(int))/n
    cooc.index = [c.replace('↑','') for c in icols]
    cooc.columns = [c.replace('↑','') for c in icols]
    return rdf, item_sup, cooc

# ── Regression ────────────────────────────────────────────────────────────────

def run_regression(df):
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.model_selection import train_test_split, cross_val_score, KFold
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    df=df.copy()
    le=LabelEncoder(); df['Major_enc']=le.fit_transform(df['Major'])
    feats=['Age','Year_of_Study','Study_Hours_Per_Week','Assignment_Completion_Rate',
           'Average_Test_Score','Retention_Score','Learning_Speed_Index','Cognitive_Load_Score',
           'Stress_Level','Adaptive_Schedule_Adjustments','Quiz_Frequency_Adjustment',
           'Resource_Recommendations','AI_Usage_Hours_Per_Week','Productivity_Improvement',
           'Satisfaction_Score','Study_Efficiency_Index','Engagement_Level','Major_enc']
    X,y=df[feats],df['GPA_Change']
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
    sc=StandardScaler(); Xtr_s,Xte_s=sc.fit_transform(Xtr),sc.transform(Xte)

    results,preds={},{}
    for name,model in [('Linear Regression',LinearRegression()),
                        ('Ridge (α=1)',Ridge(alpha=1.0)),
                        ('Lasso (α=0.01)',Lasso(alpha=0.01,max_iter=5000))]:
        model.fit(Xtr_s,ytr); yp=model.predict(Xte_s)
        preds[name]=yp
        cv=cross_val_score(model,Xtr_s,ytr,cv=KFold(5),scoring='r2').mean()
        results[name]=dict(r2=round(r2_score(yte,yp),4),
                           mae=round(mean_absolute_error(yte,yp),4),
                           rmse=round(np.sqrt(mean_squared_error(yte,yp)),4),
                           cv_r2=round(cv,4))

    best=LinearRegression(); best.fit(Xtr_s,ytr)
    coef_df=pd.DataFrame({'feature':feats,'coefficient':best.coef_}).sort_values('coefficient',key=abs,ascending=False)
    key=['GPA_Change','Engagement_Level','Productivity_Improvement','Stress_Level',
         'Retention_Score','Study_Efficiency_Index','AI_Usage_Hours_Per_Week','Average_Test_Score','Satisfaction_Score']
    return dict(results=results, y_test=yte, y_pred=preds['Linear Regression'],
                residuals=yte.values-preds['Linear Regression'],
                coef_df=coef_df, features=feats, corr=df[key].corr())
