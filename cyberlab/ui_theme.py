"""Shared white Streamlit shell and readable Plotly styling."""
from __future__ import annotations

PALETTE=['#2563eb','#0f766e','#b45309','#7c3aed','#be123c','#475569','#0284c7']
APP_CSS='''<style>
:root {color-scheme:light; --lab-ink:#0f172a; --lab-muted:#475569; --lab-line:#e2e8f0; --lab-accent:#2563eb;}
.stApp {background:#ffffff;color:var(--lab-ink);}
[data-testid="stHeader"] {background:rgba(255,255,255,.96);border-bottom:1px solid #f1f5f9;}
[data-testid="stSidebar"] {background:#f8fafc;border-right:1px solid var(--lab-line);}
[data-testid="stSidebar"] h2 {font-size:1.18rem;letter-spacing:-.025em;color:var(--lab-ink);}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {font-size:.77rem;line-height:1.55;color:var(--lab-muted);}
[data-testid="stSidebar"] [role="radiogroup"][aria-label="Workspace page"] {display:flex;flex-direction:column;flex-wrap:nowrap!important;max-height:52vh;overflow-y:auto;overflow-x:hidden;scrollbar-width:thin;padding-right:6px;}
[data-testid="stSidebar"] [role="radiogroup"][aria-label="Workspace page"] > div {flex-shrink:0;width:100%;}
[data-testid="stSidebar"] [role="radiogroup"][aria-label="Workspace page"] label {width:100%;}
[data-testid="stSidebar"] [role="radiogroup"][aria-label="Data mode"] {display:flex;flex-flow:row nowrap!important;gap:3px;width:100%;}
[data-testid="stSidebar"] [role="radiogroup"][aria-label="Data mode"] button {flex:1;min-width:0;padding:7px 5px;font-size:.77rem;}
[data-testid="stSidebar"] [role="radiogroup"][aria-label="Workflow"] {gap:.35rem;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {padding:3px 0;}
[data-testid="stMetric"] {background:#ffffff;border:1px solid var(--lab-line);border-radius:12px;padding:18px 20px;box-shadow:0 2px 5px rgba(15,23,42,.035);}
[data-testid="stMetricLabel"] {color:var(--lab-muted);font-size:.82rem;}
[data-testid="stMetricValue"] {color:var(--lab-ink);font-size:1.85rem;font-weight:650;letter-spacing:-.035em;}
[data-testid="stPlotlyChart"] {background:#ffffff;border:1px solid var(--lab-line);border-radius:14px;padding:8px;}
[data-testid="stExpander"] {background:#ffffff;border-color:var(--lab-line);border-radius:10px;}
[data-testid="stDataFrame"] {border:1px solid var(--lab-line);border-radius:10px;overflow:hidden;}
[data-testid="stCaptionContainer"] {color:var(--lab-muted);line-height:1.6;}
[data-testid="stCaptionContainer"] p {color:var(--lab-muted);}
h1,h2,h3 {color:var(--lab-ink);letter-spacing:-.025em;line-height:1.22;}
h1 {font-size:2.1rem;font-weight:700;padding-bottom:.4rem;}
h2 {font-size:1.45rem;margin-top:.65rem;}
h3 {font-size:1.12rem;}
p,li {line-height:1.65;}
a {color:#1d4ed8;}
hr {border-color:var(--lab-line);}
.block-container {padding-top:2.4rem;padding-bottom:3rem;max-width:1440px;}
.stButton>button,.stDownloadButton>button {border-radius:8px;font-weight:550;}
button:focus-visible,a:focus-visible,input:focus-visible {outline:2px solid var(--lab-accent);outline-offset:3px;}
@media(max-width:768px) {
 .block-container {padding:1.3rem 1rem 2rem;}
 h1 {font-size:1.75rem;}
 [data-testid="stMetric"] {padding:14px;}
 [data-testid="stMetricValue"] {font-size:1.5rem;}
 [data-testid="stPlotlyChart"] {padding:2px;}
}
</style>'''


def style_chart(fig):
    fig.update_layout(template='plotly_white',paper_bgcolor='#ffffff',plot_bgcolor='#ffffff',
                      font={'color':'#0f172a','family':'Arial, sans-serif','size':12},
                      colorway=PALETTE,margin={'t':60,'b':50,'l':40,'r':25},
                      legend={'orientation':'h','y':-.22},hoverlabel={'bgcolor':'#ffffff','font_color':'#0f172a'})
    fig.update_xaxes(gridcolor='#e2e8f0',zerolinecolor='#cbd5e1',automargin=True)
    fig.update_yaxes(gridcolor='#e2e8f0',zerolinecolor='#cbd5e1',automargin=True)
    return fig


def apply_theme():
    import plotly.graph_objects as go
    import plotly.io as pio
    import streamlit as st
    pio.templates['cyberlab_light']=go.layout.Template(layout=style_chart(go.Figure()).layout)
    pio.templates.default='cyberlab_light'
    st.markdown(APP_CSS,unsafe_allow_html=True)
