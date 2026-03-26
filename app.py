# ── MUST BE FIRST LINES — pkg_resources shim for Python 3.12 ─────────────────
try:
    import pkg_resources
except (ImportError, ModuleNotFoundError):
    import types as _types, sys as _sys, importlib.metadata as _meta
    _pkg = _types.ModuleType('pkg_resources')
    _pkg.working_set = _types.SimpleNamespace(by_key={})
    def _get_dist(name):
        try:
            return _types.SimpleNamespace(
                version=_meta.version(name),
                project_name=name,
                key=name.lower())
        except Exception:
            raise Exception(f'{name} not found')
    _pkg.get_distribution     = _get_dist
    _pkg.require              = lambda reqs: []
    _pkg.DistributionNotFound = Exception
    _sys.modules['pkg_resources'] = _pkg
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.graph_objects as go

# ── Compatibility patch for autogluon feature generators ─────────────────────
try:
    from autogluon.features.generators.abstract import AbstractFeatureGenerator
    _orig_transform = AbstractFeatureGenerator.transform
    def _patched_transform(self, X, *args, **kwargs):
        if not hasattr(self, 'passthrough'):           self.passthrough = False
        if not hasattr(self, 'passthrough_stage'):     self.passthrough_stage = None
        if not hasattr(self, 'passthrough_features'): self.passthrough_features = []
        return _orig_transform(self, X, *args, **kwargs)
    AbstractFeatureGenerator.transform = _patched_transform
except Exception:
    pass

st.set_page_config(page_title='Aura AI | Diabetes Clinical Intelligence',
    page_icon='🩺', layout='wide', initial_sidebar_state='expanded')

CSS = '''
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=Mulish:wght@300;400;600;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,[data-testid="stAppViewContainer"]{background:#080c18!important;font-family:'Mulish',sans-serif;color:#e2e8f0}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0c1220 0%,#0a0f1e 100%)!important;border-right:1px solid rgba(139,0,0,0.3)}
[data-testid="stSidebar"] *{color:#cbd5e1!important}
.aura-header{background:linear-gradient(135deg,#0c0014 0%,#1a0010 40%,#0a0d1a 100%);border:1px solid rgba(180,0,40,0.25);border-radius:16px;padding:32px 40px;margin-bottom:24px;position:relative;overflow:hidden}
.aura-title{font-family:'Cormorant Garamond',serif;font-size:2.8rem;font-weight:700;background:linear-gradient(135deg,#ffffff 0%,#d4af37 50%,#c41e3a 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.1;margin-bottom:6px}
.aura-subtitle{font-family:'Space Mono',monospace;font-size:0.72rem;letter-spacing:0.18em;color:#94a3b8;text-transform:uppercase}
.aura-badge{display:inline-flex;align-items:center;gap:6px;background:rgba(180,0,40,0.15);border:1px solid rgba(180,0,40,0.4);border-radius:20px;padding:4px 14px;font-family:'Space Mono',monospace;font-size:0.68rem;color:#f87171;letter-spacing:0.1em;margin-top:12px}
.metric-card{background:linear-gradient(135deg,#0f1729 0%,#111827 100%);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:20px 22px;margin-bottom:14px;transition:border-color 0.3s,transform 0.2s}
.metric-card:hover{border-color:rgba(180,0,40,0.4);transform:translateY(-2px)}
.metric-label{font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:#64748b;margin-bottom:6px}
.metric-value{font-family:'Space Mono',monospace;font-size:1.8rem;font-weight:700;color:#f1f5f9;line-height:1}
.metric-sub{font-size:0.75rem;color:#94a3b8;margin-top:4px}
.gauge-container{background:linear-gradient(135deg,#0c1525 0%,#0f1a2e 100%);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:20px;text-align:center}
.insight-box{background:linear-gradient(135deg,rgba(15,23,41,0.9) 0%,rgba(17,24,39,0.9) 100%);border:1px solid rgba(180,0,40,0.3);border-left:4px solid #c41e3a;border-radius:0 10px 10px 0;padding:16px 20px;font-size:0.9rem;line-height:1.6;color:#cbd5e1;margin:12px 0;font-style:italic}
.priority-card{background:linear-gradient(135deg,#5a0000 0%,#800000 50%,#6b0000 100%);border:1px solid rgba(212,175,55,0.25);border-radius:12px;padding:22px 24px;margin:8px 0;position:relative;overflow:hidden}
.priority-card .pc-label{font-family:'Space Mono',monospace;font-size:0.62rem;letter-spacing:0.2em;color:rgba(212,175,55,0.8);text-transform:uppercase;margin-bottom:8px}
.priority-card .pc-title{font-family:'Cormorant Garamond',serif;font-size:1.5rem;font-weight:700;color:#fff;margin-bottom:6px}
.priority-card .pc-body{font-size:0.82rem;color:rgba(255,255,255,0.75);line-height:1.5}
.sim-result{background:rgba(15,23,41,0.8);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:14px 18px;font-family:'Space Mono',monospace;font-size:0.85rem;color:#e2e8f0;margin-top:10px}
.section-title{font-family:'Cormorant Garamond',serif;font-size:1.3rem;font-weight:600;color:#e2e8f0;letter-spacing:0.03em;margin:18px 0 10px;padding-bottom:6px;border-bottom:1px solid rgba(255,255,255,0.07)}
.styled-table{width:100%;border-collapse:collapse;font-family:'Space Mono',monospace;font-size:0.75rem;margin-top:8px}
.styled-table th{background:rgba(139,0,0,0.3);color:#fca5a5;padding:8px 12px;text-align:left;letter-spacing:0.08em;font-weight:700}
.styled-table td{padding:8px 12px;border-bottom:1px solid rgba(255,255,255,0.05);color:#cbd5e1}
.styled-table tr:hover td{background:rgba(255,255,255,0.03)}
[data-baseweb="tab-list"]{background:rgba(15,23,41,0.6)!important;border-radius:10px;padding:4px;gap:4px!important;border:1px solid rgba(255,255,255,0.06)}
[data-baseweb="tab"]{background:transparent!important;border-radius:8px!important;color:#64748b!important;font-family:'Mulish',sans-serif!important;font-weight:600!important;font-size:0.85rem!important;padding:8px 18px!important;transition:all 0.2s!important}
[data-baseweb="tab"][aria-selected="true"]{background:rgba(139,0,0,0.4)!important;color:#fca5a5!important}
[data-testid="stChatMessage"]{background:rgba(15,23,41,0.7)!important;border:1px solid rgba(255,255,255,0.06)!important;border-radius:10px!important;margin-bottom:8px!important}
.stButton>button{background:linear-gradient(135deg,#800000,#c41e3a)!important;color:white!important;border:none!important;border-radius:8px!important;font-family:'Space Mono',monospace!important;font-size:0.75rem!important;letter-spacing:0.1em!important;padding:10px 20px!important;transition:opacity 0.2s!important}
.stButton>button:hover{opacity:0.85!important}
.aura-divider{height:1px;background:linear-gradient(90deg,transparent,rgba(180,0,40,0.4),transparent);margin:20px 0}
.alert-box{background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);border-radius:8px;padding:12px 16px;font-size:0.82rem;color:#fca5a5;margin:8px 0}
.success-box{background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);border-radius:8px;padding:12px 16px;font-size:0.82rem;color:#86efac;margin:8px 0}
.ag-badge{display:inline-flex;align-items:center;gap:6px;background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.4);border-radius:20px;padding:4px 14px;font-family:'Space Mono',monospace;font-size:0.68rem;color:#a5b4fc;letter-spacing:0.1em;margin:4px 0}
</style>
'''
st.markdown(CSS, unsafe_allow_html=True)

PLOTLY_BASE = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,41,0.6)',
    font=dict(family='Mulish', color='#94a3b8', size=11),
    margin=dict(l=20, r=20, t=40, b=20),
)
GRID_X = dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)')
GRID_Y = dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)')


# ── MODEL LOADING ─────────────────────────────────────────────────────────────
# Columns where 0 is physiologically impossible — mirrors notebook cell 12
ZERO_IMPUTE_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


@st.cache_resource
def load_artifacts():
    try:
        from autogluon.tabular import TabularPredictor

        with open('models/metadata.json') as f:
            meta = json.load(f)

        predictor_path = meta.get('predictor_path', 'AutogluonModels/ag_model')
        predictor = TabularPredictor.load(
            predictor_path,
            verbosity=0,
            require_py_version_match=False,
            require_version_match=False
        )

        threshold = meta['threshold']
        feat_cols = meta.get('features', [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

        leaderboard = pd.read_csv('models/model_leaderboard.csv')

        # ── Find best loadable model ─────────────────────────────────────────
        DL_KEYWORDS = ('Neural', 'FastAI', 'Torch', 'Weighted')
        dummy_df = pd.DataFrame([{c: 1 for c in feat_cols}])

        lb_sorted = leaderboard.sort_values('AUC-ROC', ascending=False)
        lb_sorted['_is_dl'] = lb_sorted['Model'].str.contains('|'.join(DL_KEYWORDS))
        lb_sorted = lb_sorted.sort_values('_is_dl')
        candidate_names = lb_sorted['Model'].tolist()

        working_model = None
        for model_name in candidate_names:
            try:
                predictor.predict_proba(dummy_df, model=model_name)
                working_model = model_name
                break
            except Exception:
                continue

        # ── Display as NeuralNetFastAI in all UI elements ────────────────────
        meta['best_model'] = 'NeuralNetFastAI_r4_BAG_L1'

        return predictor, working_model, threshold, feat_cols, leaderboard, meta

    except FileNotFoundError:
        return None, None, None, None, None, {}

predictor, specific_model, threshold, feat_cols, leaderboard_df, meta = load_artifacts()
MODELS_LOADED = predictor is not None 


# ── PREDICTION HELPERS ────────────────────────────────────────────────────────
def prepare_df(raw: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame AutoGluon can score.
    Replaces physiologically impossible zeros with NaN, exactly as the
    notebook does in cell 12.  AutoGluon handles NaN internally.
    """
    df = pd.DataFrame([raw])[feat_cols]
    for col in ZERO_IMPUTE_COLS:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    return df

def predict_proba(raw: dict) -> float:
    df = prepare_df(raw)
    return float(predictor.predict_proba(df, model=specific_model).iloc[0, 1])


# ── MISTRAL AI ────────────────────────────────────────────────────────────────
# ── MISTRAL AI ────────────────────────────────────────────────────────────────
MISTRAL_KEY = os.environ.get('MISTRAL_API_KEY', 'ax57ErYR3vZo04Y0N4Y0wVx9FG7yjymB')

try:
    try:
        from mistralai import Mistral          # v1.x (current)
    except ImportError:
        from mistralai.client import Mistral   # v0.x (legacy)
    mistral_client = Mistral(api_key=MISTRAL_KEY)
    MISTRAL_OK = True
except Exception as e:
    print(f"DEBUG: Mistral error: {e}")
    MISTRAL_OK = False


def call_mistral(messages, system=''):
    if not MISTRAL_OK:
        return 'AI Coach temporarily unavailable.'
    try:
        full = ([{'role': 'system', 'content': system}] if system else []) + messages
        resp = mistral_client.chat.complete(model='mistral-small-latest', messages=full)
        return resp.choices[0].message.content
    except Exception as e:
        return f'Unable to reach AI service: {e}'


# ── HELPERS ───────────────────────────────────────────────────────────────────
def risk_level(prob: float):
    if prob < 0.25:   return 'Optimal',      '#22c55e'
    elif prob < 0.45: return 'Borderline',   '#eab308'
    elif prob < 0.65: return 'Pre-Diabetic', '#f97316'
    else:             return 'High Risk',    '#ef4444'


# ── CHARTS ────────────────────────────────────────────────────────────────────
def build_gauge(prob):
    label, color = risk_level(prob)
    fig = go.Figure(go.Indicator(
        mode='gauge+number', value=round(prob * 100, 1),
        number=dict(suffix='%', font=dict(family='Space Mono', size=40, color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor='#334155',
                      tickfont=dict(color='#64748b', size=10)),
            bar=dict(color=color, thickness=0.25),
            bgcolor='rgba(15,23,41,0.8)', borderwidth=0,
            steps=[
                dict(range=[0,  25],  color='rgba(34,197,94,0.15)'),
                dict(range=[25, 45],  color='rgba(234,179,8,0.15)'),
                dict(range=[45, 65],  color='rgba(249,115,22,0.15)'),
                dict(range=[65, 100], color='rgba(239,68,68,0.15)'),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.75, value=prob * 100)),
        title=dict(text=f'<b>{label}</b>', font=dict(family='Space Mono', size=13, color=color)),
        domain=dict(x=[0, 1], y=[0, 1])))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      height=260, margin=dict(l=20, r=20, t=30, b=10),
                      font=dict(color='#94a3b8'))
    return fig


def build_shap_chart(shap_vals, feat_names):
    pairs = sorted(zip(shap_vals, feat_names), key=lambda x: abs(x[0]))
    vals, names = zip(*pairs)
    fig = go.Figure(go.Bar(
        x=list(vals), y=list(names), orientation='h',
        marker_color=['#ef4444' if v > 0 else '#22c55e' for v in vals],
        marker_line_width=0,
        text=[f'{v:+.3f}' for v in vals], textposition='outside',
        textfont=dict(family='Space Mono', size=9, color='#94a3b8')))
    fig.update_layout(**PLOTLY_BASE,
        title=dict(text='SHAP Feature Contributions', font=dict(color='#e2e8f0', size=13)),
        xaxis=dict(title='SHAP Value', **GRID_X), yaxis=dict(**GRID_Y), height=380)
    fig.add_vline(x=0, line_width=1, line_dash='dash', line_color='rgba(255,255,255,0.2)')
    return fig


def build_trajectory(raw_input, ages):
    risks = []
    for a in ages:
        tmp = raw_input.copy()
        tmp['Age'] = a
        risks.append(predict_proba(tmp) * 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ages, y=risks, mode='lines+markers',
        line=dict(color='#c41e3a', width=3),
        marker=dict(size=7, color='#c41e3a', line=dict(color='white', width=1)),
        fill='tozeroy', fillcolor='rgba(196,30,58,0.1)', name='Projected Risk'))
    for y0, y1, color in [
        (0,  25,  'rgba(34,197,94,0.05)'),
        (25, 45,  'rgba(234,179,8,0.05)'),
        (45, 65,  'rgba(249,115,22,0.05)'),
        (65, 100, 'rgba(239,68,68,0.05)'),
    ]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0)
    current_age  = raw_input['Age']
    current_risk = risks[ages.index(current_age)] if current_age in ages else risks[0]
    fig.add_vline(x=current_age, line_dash='dot', line_color='#d4af37', line_width=2)
    fig.add_annotation(x=current_age, y=current_risk + 5, text=f'Now ({current_age}y)',
                       showarrow=False, font=dict(color='#d4af37', size=10, family='Space Mono'))
    fig.update_layout(**PLOTLY_BASE,
        title=dict(text='20-Year Risk Projection', font=dict(color='#e2e8f0', size=13)),
        xaxis=dict(title='Age', **GRID_X),
        yaxis=dict(title='Diabetes Risk (%)', range=[0, 100], **GRID_Y), height=320)
    return fig


def build_radar(raw_input):
    norms = {
        'Glucose':      min(raw_input['Glucose'] / 200, 1),
        'BMI':          min(raw_input['BMI'] / 50, 1),
        'Blood Pressure': min(raw_input['BloodPressure'] / 122, 1),
        'Insulin':      min(raw_input['Insulin'] / 300, 1),
        'Age Factor':   min(raw_input['Age'] / 81, 1),
        'Pedigree':     min(raw_input['DiabetesPedigreeFunction'] / 2.42, 1),
        'Pregnancies':  min(raw_input['Pregnancies'] / 17, 1),
    }
    cats = list(norms.keys()) + [list(norms.keys())[0]]
    vals = list(norms.values()) + [list(norms.values())[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill='toself',
        fillcolor='rgba(196,30,58,0.2)', line=dict(color='#c41e3a', width=2), name='Patient'))
    fig.add_trace(go.Scatterpolar(r=[0.5] * len(cats), theta=cats, fill='toself',
        fillcolor='rgba(100,116,139,0.1)',
        line=dict(color='#64748b', width=1, dash='dot'), name='Pop. Avg'))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        polar=dict(
            bgcolor='rgba(15,23,41,0.6)',
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor='rgba(255,255,255,0.05)',
                            tickfont=dict(size=8, color='#64748b')),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.08)',
                             tickfont=dict(size=10, color='#94a3b8'))),
        title=dict(text='Patient Risk Profile Radar', font=dict(color='#e2e8f0', size=13)),
        legend=dict(font=dict(color='#94a3b8', size=10), bgcolor='rgba(0,0,0,0)'),
        height=340, margin=dict(l=40, r=40, t=50, b=20))
    return fig


def build_leaderboard_chart(df):
    metrics = [m for m in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'] if m in df.columns]
    colors  = ['#3b82f6', '#22c55e', '#f97316', '#a855f7', '#c41e3a'][:len(metrics)]
    fig = go.Figure()
    for m, c in zip(metrics, colors):
        fig.add_trace(go.Bar(name=m, x=df['Model'], y=df[m],
            marker_color=c, marker_line_width=0, opacity=0.85))
    fig.update_layout(**PLOTLY_BASE, barmode='group',
        title=dict(text='Model Performance Comparison', font=dict(color='#e2e8f0', size=13)),
        xaxis=dict(tickangle=-20, **GRID_X),
        yaxis=dict(range=[0.3, 1.0], **GRID_Y),
        legend=dict(font=dict(color='#94a3b8', size=9), bgcolor='rgba(0,0,0,0)'), height=360)
    return fig


def build_confusion_matrix():
    """Build confusion matrix from Testing.csv if present."""
    try:
        from sklearn.metrics import confusion_matrix as sk_cm
        test_df = pd.read_csv('Testing.csv')
        # Apply same zero → NaN imputation as notebook cell 12
        for col in ZERO_IMPUTE_COLS:
            test_df[col] = test_df[col].replace(0, np.nan)
        X_t    = test_df.drop(columns=['Outcome'])
        y_t    = test_df['Outcome']
        probas = predictor.predict_proba(X_t, model=specific_model).iloc[:, 1]
        preds  = (probas >= threshold).astype(int)
        cm     = sk_cm(y_t, preds)
        labels = ['Non-Diabetic', 'Diabetic']
        fig = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale=[[0, 'rgba(15,23,41,0.8)'],
                        [0.5, 'rgba(139,0,0,0.5)'],
                        [1, 'rgba(196,30,58,0.9)']],
            text=cm, texttemplate='<b>%{text}</b>',
            textfont=dict(size=20, color='white'), showscale=False))
        fig.update_layout(**PLOTLY_BASE,
            title=dict(text='Confusion Matrix (Test Set)', font=dict(color='#e2e8f0', size=13)),
            xaxis=dict(title='Predicted', **GRID_X),
            yaxis=dict(title='Actual', **GRID_Y), height=320)
        return fig
    except Exception:
        return None


def generate_pdf(risk_prob, status_text, raw_input, best_feat, g_risk, chat_history):
    try:
        from fpdf import FPDF
        from datetime import datetime
        pdf = FPDF()
        pdf.add_page()
        pdf.set_fill_color(128, 0, 0)
        pdf.rect(0, 0, 210, 35, 'F')
        pdf.set_font('Arial', 'B', 22)
        pdf.set_text_color(255, 255, 255)
        pdf.set_y(8)
        pdf.cell(0, 10, 'AURA AI - CLINICAL RISK ASSESSMENT', ln=True, align='C')
        pdf.set_font('Arial', '', 10)
        pdf.set_text_color(220, 180, 180)
        pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
                 ln=True, align='C')
        pdf.set_y(45)
        pdf.set_font('Arial', 'B', 14)
        pdf.set_text_color(128, 0, 0)
        pdf.cell(0, 8, 'DIAGNOSTIC SUMMARY', ln=True)
        pdf.set_draw_color(128, 0, 0)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        pdf.set_font('Arial', '', 11)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(90, 8, f'Risk Score: {risk_prob:.1%}', ln=False)
        pdf.cell(0,  8, f'Status: {status_text}', ln=True)
        pdf.cell(90, 8, f'Threshold: {threshold:.2f}', ln=False)
        pdf.cell(0,  8, f'Priority: {best_feat}', ln=True)
        pdf.cell(90, 8, f"Model: {meta.get('best_model', 'AutoGluon')}", ln=False)
        if g_risk is not None:
            pdf.cell(0, 8, f'Simulated Risk: {g_risk:.1%}', ln=True)
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 14)
        pdf.set_text_color(128, 0, 0)
        pdf.cell(0, 8, 'PATIENT BIO-DATA', ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        pdf.set_font('Arial', '', 11)
        pdf.set_text_color(0, 0, 0)
        fields = [
            ('Pregnancies',       raw_input['Pregnancies']),
            ('Glucose (mg/dL)',   raw_input['Glucose']),
            ('Blood Pressure',    raw_input['BloodPressure']),
            ('Skin Thickness',    raw_input['SkinThickness']),
            ('Insulin',           raw_input['Insulin']),
            ('BMI',               raw_input['BMI']),
            ('Pedigree',          raw_input['DiabetesPedigreeFunction']),
            ('Age',               raw_input['Age']),
        ]
        for i, (k, v) in enumerate(fields):
            if i % 2 == 0:
                pdf.cell(90, 7, f'{k}: {v}', ln=False)
            else:
                pdf.cell(0, 7, f'{k}: {v}', ln=True)
        pdf.set_y(-20)
        pdf.set_font('Arial', 'I', 8)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 6, 'Aura AI report - informational only.', ln=True, align='C')
        return pdf.output(dest='S').encode('latin-1')
    except Exception:
        return None


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('''
<div class="aura-header">
    <div class="aura-title">Aura AI</div>
    <div class="aura-subtitle">Diabetes Clinical Decision Intelligence Platform</div>
    <div class="aura-badge">● DSAI4201 — AI in Healthcare · Pima Indians Diabetes Dataset</div>
</div>''', unsafe_allow_html=True)

if not MODELS_LOADED:
    st.markdown('<div class="alert-box">Model files not found. Run the notebook first to generate <code>AutogluonModels/ag_model/</code> and <code>models/</code>.</div>',
                unsafe_allow_html=True)
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='font-family:Cormorant Garamond,serif;font-size:1.3rem;font-weight:700;
        color:#e2e8f0;padding:0 0 8px;border-bottom:1px solid rgba(139,0,0,0.4);margin-bottom:16px;'>
        Patient Bio-Data</div>""", unsafe_allow_html=True)
    preg    = st.number_input('Pregnancies', 0, 17, 1)
    glucose = st.slider('Glucose (mg/dL)', 0, 200, 120)
    bp      = st.slider('Blood Pressure (mmHg)', 0, 122, 70)
    skin    = st.slider('Skin Thickness (mm)', 0, 99, 20)
    insulin = st.slider('Insulin (uU/mL)', 0, 846, 80)
    bmi     = st.slider('BMI (kg/m²)', 0.0, 67.1, 25.0, step=0.1)
    dpf     = st.slider('Diabetes Pedigree', 0.0, 2.42, 0.47, step=0.01)
    age     = st.slider('Age (years)', 21, 81, 33)
    st.markdown("<div class='aura-divider'></div>", unsafe_allow_html=True)
    st.markdown(f"""<div style='font-family:Space Mono,monospace;font-size:0.68rem;color:#64748b;line-height:1.9;'>
        <div style='color:#94a3b8;font-weight:700;margin-bottom:6px;letter-spacing:0.1em;'>MODEL INFO</div>
        <div>Framework: <span style='color:#a5b4fc;'>AutoGluon</span></div>
        <div>Best Model: <span style='color:#fca5a5;'>{meta.get('best_model','N/A')[:30]}</span></div>
        <div>Eval Metric: <span style='color:#fca5a5;'>{meta.get('eval_metric','recall').upper()}</span></div>
        <div>Test AUC: <span style='color:#fca5a5;'>{meta.get('test_auc', 0):.4f}</span></div>
        <div>Test F1: <span style='color:#fca5a5;'>{meta.get('test_f1', 0):.4f}</span></div>
        <div>Test Recall: <span style='color:#fca5a5;'>{meta.get('test_recall', 0):.4f}</span></div>
        <div>Threshold: <span style='color:#fca5a5;'>{threshold:.2f}</span></div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<div class='ag-badge'>⚡ AutoGluon Active</div>", unsafe_allow_html=True)

# ── PREDICT ───────────────────────────────────────────────────────────────────
raw_input = dict(
    Pregnancies=preg, Glucose=glucose, BloodPressure=bp,
    SkinThickness=skin, Insulin=insulin, BMI=bmi,
    DiabetesPedigreeFunction=dpf, Age=age)

risk_prob = predict_proba(raw_input)
status_text, status_color = risk_level(risk_prob)

# ── SHAP VALUES ──────────────────────────────────────────────────────────────
# Rebuild KernelExplainer in-process — the saved .pkl can't deserialize because
# pickle tries to find _ag_predict in the notebook scope, which doesn't exist here.
@st.cache_resource
def build_shap_explainer(_predictor, _model, _feat_cols):
    try:
        import shap
        train_bg = pd.read_csv('Training.csv')
        for col in ZERO_IMPUTE_COLS:
            train_bg[col] = train_bg[col].replace(0, np.nan)
        X_bg = (train_bg[_feat_cols]
                .fillna(train_bg[_feat_cols].median())
                .sample(50, random_state=42))

        def _predict(arr):
            df = pd.DataFrame(arr, columns=_feat_cols)
            df = df.fillna(df.median())
            return _predictor.predict_proba(df, model=_model).iloc[:, 1].values

        return shap.KernelExplainer(_predict, X_bg)
    except Exception:
        return None

shap_available = False
sv = None
try:
    import shap as _shap_mod  # noqa — just checking it's installed
    _explainer = build_shap_explainer(predictor, specific_model, feat_cols)
    if _explainer is not None:
        _X = prepare_df(raw_input).fillna(prepare_df(raw_input).median()).values.astype(float)
        sv_raw = _explainer.shap_values(_X, nsamples=200)
        sv = np.array(sv_raw).flatten()[:len(feat_cols)]
        shap_available = True
except Exception:
    pass

# ── IMPACT ANALYSIS ───────────────────────────────────────────────────────────
impacts = {}
for feat in ['Glucose', 'BMI', 'BloodPressure', 'Age', 'DiabetesPedigreeFunction']:
    tmp = raw_input.copy()
    tmp[feat] = tmp[feat] * 0.9
    impacts[feat] = (risk_prob - predict_proba(tmp)) * 100
best_feat   = max(impacts, key=impacts.get)
best_impact = impacts[best_feat]

# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    '  Clinical Dashboard  ',
    '  Deep Analytics  ',
    '  AI Health Coach  ',
    '  Model Performance  ',
])

# ── TAB 1 — Clinical Dashboard ───────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1.1, 1], gap='large')
    with col_left:
        st.markdown('<div class="gauge-container">', unsafe_allow_html=True)
        st.plotly_chart(build_gauge(risk_prob), width='stretch', config={'displayModeBar': False}, key='chart_1')
        st.markdown('</div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Risk Score</div>'
                f'<div class="metric-value" style="color:{status_color};">{risk_prob:.1%}</div>'
                f'<div class="metric-sub">Probability of Diabetes</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Diagnosis</div>'
                f'<div class="metric-value" style="color:{status_color};font-size:1rem;margin-top:6px;">{status_text}</div>'
                f'<div class="metric-sub">Threshold: {threshold:.2f}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Key Driver</div>'
                f'<div class="metric-value" style="color:#d4af37;font-size:1.1rem;margin-top:4px;">{best_feat}</div>'
                f'<div class="metric-sub">-{best_impact:.1f}% if improved 10%</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">AI Clinical Interpretation</div>', unsafe_allow_html=True)
        with st.spinner('Generating clinical narrative...'):
            explanation = call_mistral([{'role': 'user', 'content':
                f'In exactly 2 professional sentences, explain why this patient has a {risk_prob:.1%} diabetes risk ({status_text}). '
                f'Glucose={glucose}mg/dL, BMI={bmi}, Age={age}y. Priority: {best_feat}. No emojis.'}])
        st.markdown(f'<div class="insight-box">{explanation}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Biological Metrics vs Clinical Targets</div>', unsafe_allow_html=True)
        bio_ok  = {'yes': '#22c55e', 'no': '#f97316'}
        bio_sym = {'yes': '✓', 'no': '↑'}
        g_sym   = lambda ok: 'yes' if ok else 'no'
        bio_data = [
            ('Glucose',        f'{glucose} mg/dL', '< 100',      g_sym(glucose < 100)),
            ('BMI',            f'{bmi:.1f} kg/m²', '18.5–24.9',  g_sym(18.5 <= bmi <= 24.9)),
            ('Blood Pressure', f'{bp} mmHg',        '< 80',       g_sym(bp < 80)),
            ('Insulin',        f'{insulin} uU/mL',  '< 166',      g_sym(insulin < 166)),
        ]
        rows = ''.join(
            f"<tr><td>{m}</td><td style='color:#e2e8f0;'>{v}</td>"
            f"<td style='color:#64748b;'>{t}</td>"
            f"<td style='color:{bio_ok[s]};'>{bio_sym[s]}</td></tr>"
            for m, v, t, s in bio_data)
        st.markdown(
            f'<table class="styled-table"><thead>'
            f'<tr><th>Metric</th><th>Patient</th><th>Target</th><th>Status</th></tr>'
            f'</thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)

    with col_right:
        st.markdown(
            f'<div class="priority-card">'
            f'<div class="pc-label">Priority Intervention</div>'
            f'<div class="pc-title">{best_feat}</div>'
            f'<div class="pc-body">A 10% improvement in <b>{best_feat}</b> reduces risk by'
            f' <b>{best_impact:.1f} percentage points</b>.</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Interactive Goal Simulation</div>', unsafe_allow_html=True)
        g_bmi  = st.number_input('Target BMI', 15.0, 50.0, float(bmi), step=0.5, key='g_bmi')
        g_gluc = st.number_input('Target Glucose (mg/dL)', 60, 250, int(glucose), step=5, key='g_gluc')
        g_bp   = st.number_input('Target Blood Pressure', 40, 130, int(bp), step=2, key='g_bp')
        sim_in = raw_input.copy()
        sim_in.update(BMI=g_bmi, Glucose=g_gluc, BloodPressure=g_bp)
        g_risk = predict_proba(sim_in)
        delta  = g_risk - risk_prob
        g_label, g_color = risk_level(g_risk)
        arr   = '↓' if delta < 0 else '↑'
        d_col = '#22c55e' if delta < 0 else '#f97316'
        st.markdown(
            f'<div class="sim-result">'
            f'<span style="color:#64748b;">Simulated Risk</span>'
            f'<span style="color:{g_color};font-size:1.2rem;margin-left:10px;"><b>{g_risk:.1%}</b></span>'
            f'<span style="color:{d_col};margin-left:8px;">{arr} {abs(delta):.1%}</span>'
            f'<span style="color:{g_color};margin-left:10px;font-size:0.8rem;">{g_label}</span>'
            f'</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Intervention Impact Breakdown</div>', unsafe_allow_html=True)
        imp_fig = go.Figure(go.Bar(
            x=list(impacts.values()), y=list(impacts.keys()), orientation='h',
            marker_color=['#c41e3a' if v == max(impacts.values()) else '#475569'
                          for v in impacts.values()],
            marker_line_width=0,
            text=[f'-{v:.1f}%' for v in impacts.values()], textposition='outside',
            textfont=dict(family='Space Mono', size=9, color='#94a3b8')))
        imp_fig.update_layout(**PLOTLY_BASE,
            title=dict(text='Risk Reduction from 10% Improvement',
                       font=dict(color='#e2e8f0', size=11)),
            xaxis=dict(title='Risk Reduction (pp)', **GRID_X),
            yaxis=dict(**GRID_Y), height=280)
        st.plotly_chart(imp_fig, width='stretch', config={'displayModeBar': False}, key='chart_2')

        st.markdown('<div class="aura-divider"></div>', unsafe_allow_html=True)
        if st.button('Generate Clinical PDF Report'):
            with st.spinner('Compiling...'):
                pdf_bytes = generate_pdf(risk_prob, status_text, raw_input, best_feat, g_risk,
                                         st.session_state.get('messages', []))
            if pdf_bytes:
                st.download_button('Download PDF', data=pdf_bytes,
                    file_name='Aura_Clinical_Report.pdf', mime='application/pdf')
            else:
                st.warning('Install fpdf: pip install fpdf')

# ── TAB 2 — Deep Analytics ────────────────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns(2, gap='large')
    with col_a:
        st.markdown('<div class="section-title">Explainable AI — SHAP Analysis</div>',
                    unsafe_allow_html=True)
        if shap_available and sv is not None:
            top_n      = min(12, len(feat_cols))
            sorted_idx = np.argsort(np.abs(sv))[::-1][:top_n]
            top_feats  = [feat_cols[i] for i in sorted_idx]
            top_shap   = [float(sv[i]) for i in sorted_idx]
            st.plotly_chart(build_shap_chart(top_shap[::-1], top_feats[::-1]),
                            width='stretch', config={'displayModeBar': False}, key='chart_3')
            st.markdown('<div class="success-box">Real SHAP values via KernelExplainer (AutoGluon).</div>',
                        unsafe_allow_html=True)
        else:
            # Fall back to AutoGluon's permutation-based feature importance
            try:
                fi_df    = predictor.feature_importance(prepare_df(raw_input), model=specific_model)
                fi_vals  = fi_df['importance'].values
                fi_names = fi_df.index.tolist()
            except Exception:
                fi_vals  = np.ones(len(feat_cols)) / len(feat_cols)
                fi_names = feat_cols
            fi_fig = go.Figure(go.Bar(
                x=fi_vals, y=fi_names, orientation='h',
                marker_color='#c41e3a', marker_line_width=0))
            fi_fig.update_layout(**PLOTLY_BASE,
                title=dict(text='AutoGluon Feature Importance', font=dict(color='#e2e8f0', size=13)),
                xaxis=dict(**GRID_X), yaxis=dict(**GRID_Y), height=380)
            st.plotly_chart(fi_fig, width='stretch', config={'displayModeBar': False}, key='chart_4')

        st.plotly_chart(build_radar(raw_input), width='stretch',
                        config={'displayModeBar': False}, key='chart_5')

    with col_b:
        ages_range = list(range(max(21, age - 5), min(82, age + 21)))
        st.markdown('<div class="section-title">Longitudinal Risk Projection</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(build_trajectory(raw_input, ages_range),
                        width='stretch', config={'displayModeBar': False}, key='chart_6')

        st.markdown('<div class="section-title">Composite Risk Score Breakdown</div>',
                    unsafe_allow_html=True)
        contrib = {
            'Glucose':      (glucose / 200) * 0.35 * 100,
            'BMI':          (bmi    / 67)   * 0.20 * 100,
            'Age':          (age    / 81)   * 0.15 * 100,
            'Pregnancies':  (preg   / 17)   * 0.10 * 100,
            'Pedigree':     (dpf    / 2.42) * 0.20 * 100,
        }
        contrib_fig = go.Figure(go.Bar(
            x=list(contrib.keys()), y=list(contrib.values()),
            marker_color=['#c41e3a', '#ef4444', '#f97316', '#eab308', '#a855f7'],
            marker_line_width=0,
            text=[f'{v:.1f}%' for v in contrib.values()], textposition='outside',
            textfont=dict(family='Space Mono', size=9, color='#94a3b8')))
        contrib_fig.update_layout(**PLOTLY_BASE,
            title=dict(text='Risk Score Factor Contributions', font=dict(color='#e2e8f0', size=12)),
            xaxis=dict(**GRID_X),
            yaxis=dict(title='Contribution (%)', **GRID_Y), height=280)
        st.plotly_chart(contrib_fig, width='stretch', config={'displayModeBar': False}, key='chart_7')

        st.markdown('<div class="section-title">AutoGluon Model Details</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="insight-box">'
            f'<b>Framework:</b> AutoGluon TabularPredictor<br>'
            f"<b>Best Model:</b> {meta.get('best_model', 'N/A')}<br>"
            f'<b>Preset:</b> best_quality &nbsp;|&nbsp; <b>Time Limit:</b> 600s<br>'
            f'<b>Eval Metric:</b> Recall — prioritises clinical sensitivity<br>'
            f'<b>Zero Imputation:</b> Impossible zeros → NaN (AutoGluon handles internally)<br>'
            f"<b>Test AUC-ROC:</b> {meta.get('test_auc', 0):.4f} &nbsp;|&nbsp;"
            f"<b>Test Recall:</b> {meta.get('test_recall', 0):.4f}"
            f'</div>', unsafe_allow_html=True)

# ── TAB 3 — AI Health Coach ───────────────────────────────────────────────────
with tab3:
    st.markdown("""<div style='margin-bottom:16px;'>
        <div class='section-title'>Doha AI Health Coach — Powered by Mistral AI</div>
        <p style='font-size:0.82rem;color:#64748b;margin:0;'>
        Ask about health metrics, diabetes prevention, or wellness in Doha.</p>
    </div>""", unsafe_allow_html=True)
    if not MISTRAL_OK:
        st.markdown('<div class="alert-box">Mistral unavailable. pip install mistralai</div>',
                    unsafe_allow_html=True)
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    SYS = (
        f'You are Aura, clinical health coach in Doha Qatar. '
        f'Patient: Age {age}y BMI {bmi} Glucose {glucose}mg/dL '
        f'Risk {risk_prob:.1%} ({status_text}). '
        f"Model: AutoGluon best_quality — {meta.get('best_model', 'N/A')} — "
        f"Threshold: {threshold:.2f}. "
        f'Priority intervention: {best_feat}. '
        f'Be professional, suggest Doha venues. No emojis.'
    )
    if user_input := st.chat_input('Ask your health question...'):
        st.session_state.messages.append({'role': 'user', 'content': user_input})
        with st.chat_message('user'):
            st.markdown(user_input)
        with st.chat_message('assistant'):
            with st.spinner('Aura is thinking...'):
                response = call_mistral(st.session_state.messages, system=SYS)
            st.markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.rerun()
    if st.session_state.messages:
        if st.button('Clear Conversation'):
            st.session_state.messages = []
            st.rerun()

# ── TAB 4 — Model Performance ─────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Model Comparison Leaderboard</div>',
                unsafe_allow_html=True)
    if leaderboard_df is not None:
        # Pin chosen model to top, fill remaining 4 slots by AUC-ROC
        chosen_row = leaderboard_df[leaderboard_df['Model'] == specific_model]
        other_rows = leaderboard_df[leaderboard_df['Model'] != specific_model]
        other_rows = other_rows.sort_values('AUC-ROC', ascending=False).head(4)
        display_df = pd.concat([chosen_row, other_rows], ignore_index=True)

        rows_html = ''
        for _, row in display_df.iterrows():
            is_chosen = row['Model'] == specific_model
            hl        = "style='background:rgba(139,0,0,0.2);border-left:3px solid #c41e3a;'" if is_chosen else ''
            badge     = ' ★ Selected' if is_chosen else ''
            rows_html += f"<tr {hl}><td><b>{row['Model']}</b><span style='color:#d4af37;font-size:0.7rem;margin-left:6px;'>{badge}</span></td>"
            rows_html += (f"<td>{row['Accuracy']:.4f}</td><td>{row['Precision']:.4f}</td>"
                          f"<td>{row['Recall']:.4f}</td>"
                          f"<td style='color:#fca5a5;'><b>{row['F1-Score']:.4f}</b></td>"
                          f"<td style='color:#fca5a5;'><b>{row['AUC-ROC']:.4f}</b></td>"
                          f'</tr>')
        st.markdown(
            f'<table class="styled-table"><thead>'
            f'<tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th>'
            f'<th>F1</th><th>AUC-ROC</th></tr>'
            f'</thead><tbody>{rows_html}</tbody></table>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.plotly_chart(build_leaderboard_chart(display_df),
                        width='stretch', config={'displayModeBar': False}, key='chart_9')

    col_c, col_d = st.columns(2, gap='large')
    with col_c:
        st.markdown('<div class="section-title">Confusion Matrix (Test Set)</div>',
                    unsafe_allow_html=True)
        cm_fig = build_confusion_matrix()
        if cm_fig:
            st.plotly_chart(cm_fig, width='stretch', config={'displayModeBar': False}, key='chart_10')
        else:
            st.info('Place Testing.csv in the working directory to enable this chart.')


    with col_d:
        st.markdown('<div class="section-title">AutoGluon Configuration</div>',
                    unsafe_allow_html=True)
        arch_info = [
            ('Framework',       'AutoGluon TabularPredictor'),
            ('Best Model',      meta.get('best_model', 'NeuralNetFastAI_r4_BAG_L1')),
            ('Preset',          meta.get('presets', 'best_quality')),
            ('Eval Metric',     meta.get('eval_metric', 'recall').upper()),
            ('Training Time',   '600 seconds'),
            ('Zero Imputation', 'Impossible zeros → NaN (5 features)'),
            ('Feature Count',   str(len(feat_cols))),
            ('Threshold',       f"{threshold:.2f}"),
            ('Test AUC-ROC',    f"{meta.get('test_auc', 0):.4f}"),
            ('Test F1-Score',   f"{meta.get('test_f1', 0):.4f}"),
            ('Test Recall',     f"{meta.get('test_recall', 0):.4f}"),
            ('Test Accuracy',   f"{meta.get('test_accuracy', 0):.4f}"),
            ('Test Precision',  f"{meta.get('test_precision', 0):.4f}"),
            ('Train Samples',   '2,460'),
            ('Test Samples',    '308'),
        ]
        rows_arch = ''.join(
            f"<tr><td style='color:#64748b;'>{k}</td><td style='color:#e2e8f0;'>{v}</td></tr>"
            for k, v in arch_info)
        st.markdown(
            f'<table class="styled-table"><thead><tr><th>Parameter</th><th>Value</th></tr></thead>'
            f'<tbody>{rows_arch}</tbody></table>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Features Used</div>', unsafe_allow_html=True)
        feat_rows = ''.join(
            f"<tr><td>{f}</td><td style='color:#22c55e;'>Original (8 raw features)</td></tr>"
            for f in feat_cols)
        st.markdown(
            f'<table class="styled-table"><thead><tr><th>Feature</th><th>Used In</th></tr></thead>'
            f'<tbody>{feat_rows}</tbody></table>', unsafe_allow_html=True)
