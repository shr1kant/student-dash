"""
analytics.py — Shared ML engine for ALO Streamlit App
All heavy computation lives here. Pages just call functions and render.
"""

import pandas as pd
import numpy as np
import warnings
from itertools import combinations
warnings.filterwarnings('ignore')

# ── Data Loading & Caching ────────────────────────────────────────────────────

def load_raw():
    return pd.read_csv('data/ALO_raw.csv')

def load_clean():
    return pd.read_csv('data/ALO_cleaned.csv')

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
    for col in ['Cognitive_Load_Score', 'AI_Usage_Hours_Per_Week']:
        df[col] = df.groupby('_sb')[col].transform(lambda x: x.fillna(x.median()))
    df.drop('_sb', axis=1, inplace=True)

    df['_pq'] = pd.qcut(df['Productivity_Improvement'].rank(method='first'), q=4, labels=['Q1','Q2','Q3','Q4'])
    df['Satisfaction_Score'] = df.groupby('_pq')['Satisfaction_Score'].transform(
        lambda x: x.fillna(x.median()))
    df.drop('_pq', axis=1, inplace=True)
    df['Willingness_to_Pay'] = df['Willingness_to_Pay'].fillna(df['Willingness_to_Pay'].median())

    for col in ['Study_Hours_Per_Week', 'AI_Usage_Hours_Per_Week', 'Willingness_to_Pay']:
        df[col] = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))

    def mm(s): return (s - s.min()) / (s.max() - s.min())
    norms = {c: mm(df[c]) for c in ['Study_Hours_Per_Week','AI_Usage_Hours_Per_Week',
             'Productivity_Improvement','Retention_Score']}

    sei = ((df['Average_Test_Score']/100) * df['Assignment_Completion_Rate']) / (norms['Study_Hours_Per_Week'] + 0.01)
    df['Study_Efficiency_Index'] = (mm(sei.clip(upper=sei.quantile(0.99))) * 10).round(3)

    eng = (norms['AI_Usage_Hours_Per_Week']
           + df['Adaptive_Schedule_Adjustments']/df['Adaptive_Schedule_Adjustments'].max()
           + df['Quiz_Frequency_Adjustment']/df['Quiz_Frequency_Adjustment'].max()
           + df['Resource_Recommendations']/df['Resource_Recommendations'].max()) / 4
    df['Engagement_Level'] = (eng * 10).round(3)

    df['Learning_Improvement_Index'] = (
        mm(df['GPA_Change']) * 0.40 +
        norms['Productivity_Improvement'] * 0.35 +
        norms['Retention_Score'] * 0.25
    ) * 10
    df['Learning_Improvement_Index'] = df['Learning_Improvement_Index'].round(3)
    return df

# ── Clustering ────────────────────────────────────────────────────────────────

def run_clustering(df, k=4):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    features = ['Study_Hours_Per_Week','Average_Test_Score','Stress_Level',
                'AI_Usage_Hours_Per_Week','Engagement_Level','Study_Efficiency_Index',
                'Learning_Improvement_Index','GPA_Change','Satisfaction_Score']
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow + silhouette
    inertias, silhouettes = [], []
    for ki in range(2, 9):
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    km_final = KMeans(n_clusters=k, random_state=42, n_init=10)
    df = df.copy()
    df['Cluster'] = km_final.fit_predict(X_scaled)

    centroids = pd.DataFrame(scaler.inverse_transform(km_final.cluster_centers_), columns=features)
    eff_rank   = centroids['Study_Efficiency_Index'].rank()
    stress_rank= centroids['Stress_Level'].rank()
    gpa_rank   = centroids['GPA_Change'].rank()
    eng_rank   = centroids['Engagement_Level'].rank()

    persona_map = {}
    for i in range(k):
        if eff_rank[i] >= 3 and gpa_rank[i] >= 3:
            persona_map[i] = 'Efficient Achievers'
        elif stress_rank[i] >= 3 and eng_rank[i] >= 3:
            persona_map[i] = 'High-Stress Engagers'
        elif eng_rank[i] <= 2 and gpa_rank[i] <= 2:
            persona_map[i] = 'Passive Learners'
        else:
            persona_map[i] = 'Developing Strivers'

    df['Persona'] = df['Cluster'].map(persona_map)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]

    return df, {
        'inertias': inertias,
        'silhouettes': silhouettes,
        'k_range': list(range(2, 9)),
        'best_silhouette': silhouettes[k-2],
        'pca_variance': pca.explained_variance_ratio_.tolist(),
        'features': features,
        'centroids': centroids,
        'persona_map': persona_map,
    }

# ── Classification ────────────────────────────────────────────────────────────

def run_classification(df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  roc_auc_score, roc_curve, accuracy_score)

    df = df.copy()
    wtp_median = df['Willingness_to_Pay'].median()
    df['WTP_High'] = (df['Willingness_to_Pay'] >= wtp_median).astype(int)

    le = LabelEncoder()
    df['Major_enc'] = le.fit_transform(df['Major'])

    features = ['Age','Year_of_Study','Study_Hours_Per_Week','Assignment_Completion_Rate',
                'Average_Test_Score','Retention_Score','Learning_Speed_Index',
                'Cognitive_Load_Score','Stress_Level','Adaptive_Schedule_Adjustments',
                'Quiz_Frequency_Adjustment','Resource_Recommendations',
                'AI_Usage_Hours_Per_Week','Productivity_Improvement','Satisfaction_Score',
                'GPA_Change','Retention_Likelihood','Study_Efficiency_Index',
                'Engagement_Level','Learning_Improvement_Index','Major_enc']

    X, y = df[features], df['WTP_High']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                 random_state=42, class_weight='balanced')
    rf.fit(X_tr, y_train)
    y_pred_rf = rf.predict(X_te)
    y_prob_rf  = rf.predict_proba(X_te)[:,1]

    lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr.fit(X_tr, y_train)
    y_pred_lr = lr.predict(X_te)
    y_prob_lr  = lr.predict_proba(X_te)[:,1]

    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)

    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    return {
        'wtp_median': wtp_median,
        'rf': {'acc': accuracy_score(y_test,y_pred_rf), 'auc': roc_auc_score(y_test,y_prob_rf),
               'cm': confusion_matrix(y_test,y_pred_rf), 'fpr': fpr_rf, 'tpr': tpr_rf,
               'report': classification_report(y_test,y_pred_rf,target_names=['Low WTP','High WTP'],output_dict=True)},
        'lr': {'acc': accuracy_score(y_test,y_pred_lr), 'auc': roc_auc_score(y_test,y_prob_lr),
               'cm': confusion_matrix(y_test,y_pred_lr), 'fpr': fpr_lr, 'tpr': tpr_lr},
        'importances': importances,
        'features': features,
        'y_test': y_test,
        'df': df,
    }

# ── Association Rules ─────────────────────────────────────────────────────────

def run_association_rules(df, min_support=0.28, min_confidence=0.60, min_lift=1.10):
    bin_map = {
        'AI_Usage_Hours_Per_Week':'AI_Usage','Engagement_Level':'Engagement',
        'Stress_Level':'Stress','Average_Test_Score':'Test_Score',
        'Productivity_Improvement':'Productivity','GPA_Change':'GPA',
        'Satisfaction_Score':'Satisfaction','Study_Efficiency_Index':'Study_Efficiency',
        'Adaptive_Schedule_Adjustments':'Adaptive_Adj','Willingness_to_Pay':'WTP',
    }
    df = df.copy()
    for col, label in bin_map.items():
        df[f'{label}_High'] = (df[col] >= df[col].median()).astype(bool)

    item_cols = [f'{label}_High' for label in bin_map.values()]
    basket = df[item_cols].astype(bool)
    n = len(basket)

    freq = {}
    for item in item_cols:
        sup = basket[item].sum() / n
        if sup >= min_support:
            freq[frozenset([item])] = sup

    freq1 = list(freq.keys())
    for a, b in combinations(freq1, 2):
        c = a | b
        sup = basket[list(c)].all(axis=1).sum() / n
        if sup >= min_support:
            freq[c] = sup

    freq2 = [k for k in freq if len(k)==2]
    seen3 = set()
    for a, b in combinations(freq2, 2):
        c = a | b
        if len(c)==3 and c not in seen3:
            seen3.add(c)
            sup = basket[list(c)].all(axis=1).sum() / n
            if sup >= min_support:
                freq[c] = sup

    rules = []
    for itemset, sup_XY in freq.items():
        if len(itemset) < 2: continue
        for size in range(1, len(itemset)):
            for ant in combinations(list(itemset), size):
                ant = frozenset(ant)
                con = itemset - ant
                sup_X = freq.get(ant, basket[list(ant)].all(axis=1).sum()/n)
                if sup_X == 0: continue
                conf = sup_XY / sup_X
                sup_Y = freq.get(con, basket[list(con)].all(axis=1).sum()/n)
                lift = conf / sup_Y if sup_Y > 0 else 0
                if conf >= min_confidence and lift >= min_lift:
                    rules.append({
                        'antecedents': ', '.join(sorted([i.replace('_High','').replace('_',' ') for i in ant])),
                        'consequents': ', '.join(sorted([i.replace('_High','').replace('_',' ') for i in con])),
                        'support': round(sup_XY,4),
                        'confidence': round(conf,4),
                        'lift': round(lift,4),
                    })

    rules_df = pd.DataFrame(rules).sort_values('lift', ascending=False).reset_index(drop=True) if rules else pd.DataFrame()

    single_support = {c.replace('_High','').replace('_',' '): basket[c].sum()/n for c in item_cols}

    short_labels = [c.replace('_High','').replace('_',' ') for c in item_cols]
    cooc = basket.astype(int).T.dot(basket.astype(int)) / n
    cooc.index = short_labels
    cooc.columns = short_labels

    return rules_df, single_support, cooc

# ── Regression ────────────────────────────────────────────────────────────────

def run_regression(df):
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.model_selection import train_test_split, cross_val_score, KFold
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    df = df.copy()
    le = LabelEncoder()
    df['Major_enc'] = le.fit_transform(df['Major'])

    features = ['Age','Year_of_Study','Study_Hours_Per_Week','Assignment_Completion_Rate',
                'Average_Test_Score','Retention_Score','Learning_Speed_Index',
                'Cognitive_Load_Score','Stress_Level','Adaptive_Schedule_Adjustments',
                'Quiz_Frequency_Adjustment','Resource_Recommendations',
                'AI_Usage_Hours_Per_Week','Productivity_Improvement','Satisfaction_Score',
                'Study_Efficiency_Index','Engagement_Level','Major_enc']

    X, y = df[features], df['GPA_Change']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    results = {}
    preds = {}
    for name, model in [('Linear Regression', LinearRegression()),
                         ('Ridge (α=1.0)', Ridge(alpha=1.0)),
                         ('Lasso (α=0.01)', Lasso(alpha=0.01, max_iter=5000))]:
        model.fit(X_tr, y_train)
        yp = model.predict(X_te)
        preds[name] = yp
        cv = cross_val_score(model, X_tr, y_train, cv=KFold(5), scoring='r2').mean()
        results[name] = {
            'r2': round(r2_score(y_test,yp),4),
            'mae': round(mean_absolute_error(y_test,yp),4),
            'rmse': round(np.sqrt(mean_squared_error(y_test,yp)),4),
            'cv_r2': round(cv,4)
        }

    best = LinearRegression()
    best.fit(X_tr, y_train)
    coef_df = pd.DataFrame({'feature': features, 'coefficient': best.coef_}).sort_values('coefficient', key=abs, ascending=False)

    key_vars = ['GPA_Change','Engagement_Level','Productivity_Improvement','Stress_Level',
                'Retention_Score','Study_Efficiency_Index','AI_Usage_Hours_Per_Week',
                'Average_Test_Score','Satisfaction_Score']
    corr = df[key_vars].corr()

    return {
        'results': results,
        'y_test': y_test,
        'y_pred': preds['Linear Regression'],
        'residuals': y_test.values - preds['Linear Regression'],
        'coef_df': coef_df,
        'features': features,
        'corr': corr,
    }
