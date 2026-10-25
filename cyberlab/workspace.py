"""Dataset analytics workspace inspired by the local DataIQ reference.

The published benchmark is a separate reporting surface. Interactive models
never replace its winner, thresholds or evidence files.
"""
from __future__ import annotations

from hashlib import sha256
from html import escape
import importlib
import io
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

APP_MODES = ('Beginner', 'Data Scientist', 'Business Manager', 'Presentation')
TEMPLATES = {
    'Cyber Threat Detection': ('Cybersecurity', 'Inspect label balance, grouped validation and false alerts before model promotion.'),
    'Network Traffic': ('Cybersecurity', 'Profile traffic volume, protocols, temporal groups and potential capture artifacts.'),
    'General Data Project': ('General', 'Confirm target, feature meaning, missingness and independent validation units.'),
    'Sales Analysis': ('Business', 'Explore revenue and cost drivers; predictions do not establish causal effects.'),
    'Customer Churn': ('Business', 'Evaluate retention signals, class balance and intervention costs.'),
    'Student Performance': ('Education', 'Use predictions to support learners; never automate punitive decisions.'),
    'Finance / Risk': ('Regulated', 'Review temporal leakage, fairness and governance before real financial decisions.'),
    'Health Dataset': ('Regulated', 'Educational analytics only; not diagnosis or treatment advice.'),
    'Housing Demo': ('Demo', 'Upload your own housing data; no housing dataset is bundled into this cyber project.'),
}
PAGE_GROUPS = {
    'Foundation': ('Start Here','Project Manager','Smart Auto-Setup','Data Cleaning Studio','Data Quality','Data Dictionary'),
    'Discovery': ('Overview','Explore','Compare','OLAP & Export'),
    'Modeling': ('ML Lab','Evaluation','Model Explainability','Unsupervised Lab','Reinforcement Lab','Fit Diagnostics','Prediction Page','Model Save / Load'),
    'Decisions': ('Forecast','Scenario Simulator','Business Impact','Conclusion','Domain Review'),
    'Governance': ('Executive Summary','Report Generator','Production Readiness','Experiment Tracker','Model Registry','Data Pipeline','Big Data Readiness','Code Lab','Assistant'),
}
PAGE_NAMES = tuple(page for pages in PAGE_GROUPS.values() for page in pages)
PRIMARY_PAGES = {'Start Here','Project Manager','Smart Auto-Setup','Data Cleaning Studio','Data Quality','Data Dictionary','Overview','Explore','ML Lab','Evaluation','Prediction Page','Executive Summary','Report Generator'}
PAGE_MODULES = ('workspace_data', 'workspace_ml', 'workspace_advanced', 'workspace_guidance')
CORE_PAGES = {'Start Here', 'Executive Summary', 'Report Generator'}


def page_registry():
    """One module owns each route; reject missing/duplicate routes up front."""
    owners = {page: None for page in CORE_PAGES}
    for name in PAGE_MODULES:
        module = importlib.import_module(f'cyberlab.{name}')
        for page in module.PAGES:
            if page in owners:
                raise RuntimeError(f'Duplicate workspace page ownership: {page}')
            owners[page] = module
    if set(owners) != set(PAGE_NAMES):
        raise RuntimeError(f'Workspace route mismatch: {set(owners) ^ set(PAGE_NAMES)}')
    return owners


def search_workspace(query, columns, pages=PAGE_NAMES):
    """Rank route names, workflow groups and keywords, with small typo tolerance."""
    from difflib import SequenceMatcher
    keywords={
        'Evaluation':'accuracy precision recall f1 auc average precision comparison confusion false alerts',
        'Prediction Page':'predict predictions inference single batch scoring',
        'Model Explainability':'importance coefficients drivers interpretation explain features',
        'Data Cleaning Studio':'missing outliers duplicates imputation median mean repair clean',
        'OLAP & Export':'pivot cube segment heatmap export dimensions',
        'Forecast':'future lags horizon trend time series',
        'Model Save / Load':'bundle serialization import save load persistence skops',
        'Assistant':'questions help local chat',
        'Report Generator':'markdown html download report handoff',
        'Production Readiness':'validation drift model card governance checklist',
        'Business Impact':'cost false alerts missed attacks decision',
        'Fit Diagnostics':'overfit underfit train validation gap f1',
    }
    q=query.strip().lower()
    groups={page:area.lower() for area,items in PAGE_GROUPS.items() for page in items}
    ranked=[]
    for page in pages:
        name=page.lower()
        text=f'{name} {groups.get(page,"")} {keywords.get(page,"")}'
        terms=q.split()
        exact=sum(t in text for t in terms)/max(len(terms),1)
        fuzzy=max((SequenceMatcher(None,q,word).ratio() for word in name.split()),default=0)
        if exact==1 or (len(q)>=4 and fuzzy>=0.72):
            score=exact+fuzzy+(2 if q in name else 0)
            ranked.append((score,page))
    ranked.sort(key=lambda item:(-item[0],item[1]))
    matching_columns=[str(c) for c in columns if q in str(c).lower()]
    return [page for _,page in ranked],matching_columns


def read_workspace_csv(payload, max_bytes=25_000_000, max_rows=50_000):
    """Reject oversized uploads rather than silently truncating evidence."""
    if len(payload)>max_bytes:
        raise ValueError(f'CSV size exceeds the {max_bytes:,}-byte interactive limit; use the offline pipeline.')
    if not payload:
        raise ValueError('The CSV is empty.')
    last_error = None
    for encoding in ('utf-8-sig','cp1252'):
        try:
            df=pd.read_csv(io.BytesIO(payload),encoding=encoding,nrows=max_rows+1,low_memory=False)
            break
        except UnicodeDecodeError as exc:
            last_error=exc
    else:
        raise ValueError(f'Cannot decode CSV: {last_error}')
    if len(df)>max_rows:
        raise ValueError(f'CSV has more than {max_rows:,} rows; interactive uploads are bounded. No rows were silently sampled.')
    if df.empty:
        raise ValueError('CSV contains no data rows.')
    if len(df.columns)>200:
        raise ValueError('CSV has more than 200 columns.')
    df.columns=[str(c).strip() for c in df.columns]
    if not df.columns.is_unique:
        raise ValueError('CSV headers must be unique after whitespace cleanup.')
    return df


def load_default_workspace(root):
    path=Path(root)/'report_data'/'workspace_features.csv'
    if not path.is_file():
        raise ValueError(f'Real feature snapshot missing: {path}. Upload a CSV or generate a validated feature snapshot.')
    return read_workspace_csv(path.read_bytes())


def _is_leaky_or_identifier(name):
    name=str(name).lower().strip()
    return bool(name in {'id','index','window_id','flow_id','split','time_group','label','reference_label','malicious','target','prediction','predicted','attack_family'}
                or name.startswith(('score_','flagged_','prediction_','predicted_','reference_','unnamed:'))
                or name.endswith(('_id','_ip')))


def suggest_roles(df):
    columns=list(df.columns)
    date_col=next((c for c in columns if str(c).lower() in {'timestamp','@timestamp','date','datetime','time'}),None)
    if date_col is None:
        date_col=next((c for c in columns if pd.api.types.is_datetime64_any_dtype(df[c])),None)
    preferred=('malicious','label','target','outcome','churn','price','sales','value')
    target=next((c for hint in preferred for c in columns if str(c).lower()==hint),None)
    if target is None:
        candidates=[c for c in df.select_dtypes(include=['number','bool']).columns if c!=date_col and not _is_leaky_or_identifier(c)]
        target=candidates[-1] if candidates else None
    features=[c for c in df.select_dtypes(include='number').columns
              if c not in (target,date_col) and not _is_leaky_or_identifier(c)]
    values=df[target].dropna() if target else pd.Series(dtype=float)
    task=('Classification' if target and (not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)
                or (values.nunique()<=20 and (values.empty or np.all(np.equal(np.mod(pd.to_numeric(values,errors='coerce'),1),0))))) else 'Regression')
    return {'date_col':date_col,'target':target,'features':features[:40],'task':task}


def dataset_fingerprint(df,target,date_col,features):
    h=sha256(pd.util.hash_pandas_object(df,index=True).values.tobytes())
    h.update(json.dumps({'columns':list(df.columns),'target':target,'date_col':date_col,'features':list(features)},sort_keys=True).encode())
    return h.hexdigest()


def _table_markdown(df,limit=30):
    if df.empty: return '_No evidence recorded._'
    view=df.head(limit).astype(str)
    quote=lambda value:str(value).replace('|','/').replace('\n',' ')
    lines=['| '+' | '.join(map(quote,view.columns))+' |','| '+' | '.join('---' for _ in view.columns)+' |']
    lines.extend('| '+' | '.join(map(quote,row))+' |' for row in view.itertuples(index=False,name=None))
    return '\n'.join(lines)


def workspace_report(ctx,include_numeric=True,include_dictionary=True,include_cleaning=True,include_models=True):
    """Export current project evidence, never label prior benchmark metrics as new training."""
    import streamlit as st
    df=ctx['df']
    lines=[f"# {ctx['project_name']} — analytics report",'',
           '## Project profile',f"Source: {ctx['source_name']}",f"Rows: {len(df):,}; columns: {len(df.columns)}.",
           f"Target: {ctx['target']}; date column: {ctx['date_col']}.",f"Features: {', '.join(ctx['features']) or 'None'}.",
           f"Data fingerprint: {ctx['fingerprint']}",'',
           '## Executive interpretation',
           'This interactive workspace is exploratory. Its experiments do not replace the published grouped-CV benchmark.',
           'No production readiness, causal effect, or improvement on future attacks follows from a high score alone.',
           f"Missing cells: {int(df.isna().sum().sum()):,}; duplicate rows: {int(df.duplicated().sum()):,}.",
           'Use independent capture/domain validation and preserve realistic benign/attack evaluation coverage.']
    if ctx['source_name']=='CICIDS2017 real feature snapshot':
        lines.extend(['The bundled source contains 68 positive host/minute windows from one source host and one continuous attack episode.',
                      'Downloaded full-week CICIDS2017 and CIC-DDoS2019 extensions are not silently merged into this snapshot.'])
    if ctx.get('data_scope'):
        lines.extend(['','## Input scope',ctx['data_scope']])
    if include_cleaning:
        lines+=['','## Cleaning audit',_table_markdown(pd.DataFrame(ctx['cleaning_audit']))]
    if include_models:
        result=st.session_state.get('workspace_model_result')
        lines+=['','## Workspace model evidence']
        if result is None or result.get('fingerprint')!=ctx['fingerprint']:
            lines+=['No current, matching workspace model has been trained or loaded.']
        else:
            lines+=[f"Candidate: {result.get('winner',result.get('model_name','See evaluation'))}.",
                    'Review the Evaluation, Fit Diagnostics and Registry pages; test data must not be reused to tune the winner.']
            for key in ('comparison','evaluation','metrics'):
                value=result.get(key)
                if isinstance(value,pd.DataFrame): lines.append(_table_markdown(value))
    if include_numeric:
        numeric=df.select_dtypes(include='number')
        summary=_table_markdown(numeric.describe().T.reset_index()) if len(numeric.columns) else 'No numeric columns are available in this dataset.'
        lines+=['','## Numeric summary',summary]
    if include_dictionary:
        dictionary=pd.DataFrame({'Column':df.columns,'Type':[str(x) for x in df.dtypes],
                                 'Missing':[int(df[c].isna().sum()) for c in df],
                                 'Unique':[int(df[c].nunique(dropna=True)) for c in df]})
        lines+=['','## Data dictionary',_table_markdown(dictionary,200)]
    markdown='\n\n'.join(lines)
    # Escape all dataset-derived text; no arbitrary HTML/code execution.
    html='<!doctype html><html lang="en"><head><meta charset="utf-8"><title>CyberLab analytics report</title><style>body{font:15px system-ui;max-width:1100px;margin:2rem auto;padding:1rem;background:#ffffff;color:#0f172a}pre{white-space:pre-wrap;overflow-wrap:anywhere}h1{color:#0f172a}</style></head><body><h1>CyberLab analytics evidence</h1><pre>'+escape(markdown)+'</pre></body></html>'
    return markdown,html


def default_workspace_provenance(root):
    path=Path(root)/'report_data/workspace_provenance.json'
    record=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    record.setdefault('workspace_label','CICIDS2017 real feature snapshot')
    return record


def render_workspace(project_root):
    import streamlit as st
    root=Path(project_root)
    with st.sidebar.expander('Dataset & project setup',expanded=False):
        upload=st.file_uploader('Upload CSV (25 MB / 50,000 rows maximum)',type=['csv'],key='workspace_upload')
        default_scope=default_workspace_provenance(root)
        source_name=upload.name if upload is not None else default_scope['workspace_label']
        try:
            raw=read_workspace_csv(upload.getvalue()) if upload is not None else load_default_workspace(root)
        except (ValueError,pd.errors.ParserError,pd.errors.EmptyDataError) as exc:
            st.error(str(exc));st.stop()
        signature=dataset_fingerprint(raw,None,None,[])
        if st.session_state.get('workspace_source_signature')!=signature:
            st.session_state['workspace_source_signature']=signature
            st.session_state['workspace_cleaned']=raw.copy()
            st.session_state['workspace_cleaning_audit']=[]
            st.session_state.pop('workspace_model_result',None)
        df=st.session_state['workspace_cleaned']
        project_name=st.text_input('Project name',value='Cyber Threat Intelligence',key='workspace_project_name')
        template=st.selectbox('Dataset template',list(TEMPLATES),key='workspace_template')
        mode=st.selectbox('App mode',APP_MODES,key='workspace_mode')
        st.caption(f'{TEMPLATES[template][0]} · {TEMPLATES[template][1]}')
        suggested=suggest_roles(df)
        options=['(none)',*map(str,df.columns)]
        target_choice=st.selectbox('Target column',options,index=options.index(suggested['target']) if suggested['target'] in options else 0,key='workspace_target')
        date_choice=st.selectbox('Date column',options,index=options.index(suggested['date_col']) if suggested['date_col'] in options else 0,key='workspace_date')
        target=None if target_choice=='(none)' else target_choice
        date_col=None if date_choice=='(none)' else date_choice
        allowed=[c for c in df.select_dtypes(include='number').columns if c not in (target,date_col) and not _is_leaky_or_identifier(c)]
        selected_features=st.multiselect('Numeric model features',allowed,default=allowed[:40],max_selections=40,key='workspace_features')
    ctx={'df':df,'raw_df':raw,'target':target,'date_col':date_col,'features':selected_features,
         'project_name':project_name,'source_name':source_name,'mode':mode,
         'fingerprint':dataset_fingerprint(df,target,date_col,selected_features),
         'cleaning_audit':st.session_state.get('workspace_cleaning_audit',[]),'project_root':str(root),
         'data_scope':default_scope.get('scope') if upload is None else 'User-uploaded dataset; source/label provenance requires review.'}
    previous=st.session_state.get('workspace_fingerprint')
    if previous and previous!=ctx['fingerprint']:
        st.session_state.pop('workspace_model_result',None)
    st.session_state['workspace_fingerprint']=ctx['fingerprint']
    query=st.sidebar.text_input('Search pages or columns',key='workspace_search')
    advanced=st.sidebar.checkbox('Show advanced pages',value=True,key='workspace_advanced')
    group=st.sidebar.selectbox('Workflow area',['All',*PAGE_GROUPS],key='workspace_area')
    choices=list(PAGE_NAMES if group=='All' else PAGE_GROUPS[group])
    if not advanced: choices=[p for p in choices if p in PRIMARY_PAGES]
    if query.strip():
        matches,column_matches=search_workspace(query,df.columns,choices)
        if column_matches: st.sidebar.caption('Matching columns: '+', '.join(column_matches[:12]))
        choices=matches
    if not choices:
        st.info('No pages match these filters. Clear the search or select another workflow area.');return
    page=st.sidebar.radio('Workspace page',choices,key='workspace_page')
    st.caption(f'{project_name} / {source_name} / {mode} / {len(df):,} rows')
    st.title(page)
    if source_name=='CICIDS2017 real feature snapshot':
        st.warning('Demo evidence: Thursday-only real feature windows, 68 positives from one host/episode. Additional downloaded datasets are separate; no improved detection is implied.')
    elif upload is None and default_scope.get('scope'):
        st.info(default_scope['scope'])
    if mode=='Beginner':
        st.caption('Workflow: inspect data → clean with an audit → choose independent splits → compare models → inspect mistakes → export evidence.')
    elif mode=='Business Manager':
        st.caption('Decision lens: review missed attacks, false alerts, support counts and assumptions before interpreting gains.')
    elif mode=='Presentation':
        st.caption('Presentation lens: describe the source, selected metric, held-out evidence and limits together.')
    owner = page_registry()[page]
    if owner is not None:
        owner.render(page,ctx)
    elif page=='Start Here':
        st.subheader('A complete dataset-to-evidence workflow')
        cols=st.columns(3)
        cols[0].metric('Active rows',f'{len(df):,}')
        cols[1].metric('Columns',len(df.columns))
        cols[2].metric('Available feature pages',len(PAGE_NAMES))
        for area,pages in PAGE_GROUPS.items():
            with st.expander(area,expanded=area in ('Foundation','Modeling')):
                st.write(' → '.join(pages))
        st.info('Start with Project Manager and Smart Auto-Setup. Models fit only when you press their training button. The original six benchmark views remain available by turning off Full analytics workspace.')
        st.caption('The reference housing project is untouched. No arbitrary code execution, pickle uploads or live network attack actions are provided.')
    else:
        if page=='Report Generator':
            a,b=st.columns(2)
            numeric=a.checkbox('Include numeric summary',True,key='workspace_report_numeric')
            dictionary=b.checkbox('Include data dictionary',True,key='workspace_report_dictionary')
            cleaning=a.checkbox('Include cleaning audit',True,key='workspace_report_cleaning')
            models=b.checkbox('Include current model evidence',True,key='workspace_report_models')
        else:
            numeric=dictionary=cleaning=False;models=True
            st.subheader('Dataset, decision, evidence and limitations')
        md,html=workspace_report(ctx,numeric,dictionary,cleaning,models)
        a,b,c=st.columns(3)
        a.download_button('Download workspace Markdown',md,'cyberlab_workspace_report.md','text/markdown',key='workspace_report_md')
        b.download_button('Download workspace HTML',html,'cyberlab_workspace_report.html','text/html',key='workspace_report_html')
        summary=pd.DataFrame([{'project':project_name,'source':source_name,'rows':len(df),'columns':len(df.columns),'target':target,'date':date_col,'fingerprint':ctx['fingerprint']}])
        c.download_button('Download project summary CSV',summary.to_csv(index=False),'cyberlab_project_summary.csv','text/csv',key='workspace_report_csv')
        st.markdown(md)
