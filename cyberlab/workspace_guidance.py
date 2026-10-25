"""Local evidence guidance, domain review and safe code previews.

These pages do not call an external LLM or execute user/generated Python.
"""
from __future__ import annotations

import ast
import json
import pandas as pd

PAGES = ('Domain Review', 'Code Lab', 'Assistant')


def code_preview(ctx, recipe, group=None):
    columns=list(ctx['df'].columns)
    common='import pandas as pd\n\ndf = pd.read_csv("input.csv")\n'
    if recipe=='Dataset profile':
        code=common+'print(df.shape)\nprint(df.dtypes)\nprint(df.isna().sum())\nprint(df.describe(include="all").T)\n'
    elif recipe=='Numeric correlation':
        names=list(ctx['features'])
        code=common+f'features = {names!r}\nprint(df[features].corr())\n'
    elif recipe=='Grouped summary':
        if group not in columns:
            raise ValueError('Choose an observed grouping column.')
        names=[x for x in ctx['features'] if x!=group]
        if not names:
            raise ValueError('Choose at least one numeric feature distinct from the grouping column.')
        code=common+f'features = {names!r}\nsummary = df.groupby({group!r}, dropna=False)[features].agg(["count", "mean", "median"])\nprint(summary)\nsummary.to_csv("grouped_summary.csv")\n'
    else:
        raise ValueError('Unsupported code recipe.')
    ast.parse(code)
    return code


def assistant_answer(ctx, question, model=None):
    df=ctx['df']
    question=question.strip()
    if not question or len(question)>1000:
        raise ValueError('Ask a question between 1 and 1,000 characters.')
    lines=[f"Active source: {ctx['source_name']}. Observed data: {len(df):,} rows × {len(df.columns)} columns.",
           f"Missing cells: {int(df.isna().sum().sum()):,}; duplicate rows: {int(df.duplicated().sum()):,}."]
    target=ctx.get('target')
    if target in df:
        counts=df[target].value_counts(dropna=False)
        if len(counts)<=20:
            lines.append(f"Observed target `{target}` counts: "+', '.join(f'{key!s}: {int(value):,}' for key,value in counts.items()))
        else:
            lines.append(f"Target `{target}` has {df[target].nunique():,} distinct nonmissing values; this alone does not define a classification task.")
    if model and model.get('fingerprint')==ctx['fingerprint']:
        lines.append(f"Matching session model: {model['winner']}. Selection policy: {model['cv_policy']}.")
    else:
        lines.append('No current matching workspace model is available. Train explicitly in ML Lab; no model performance can be inferred from the dataset profile.')
    q=question.lower()
    if any(term in q for term in ('attack','live','ip','detect')):
        lines.append('Use Data mode → Run data for uploaded flows/windows, or Live data for bounded Elasticsearch stream snapshots. Source IP must be observed in input; the identifier-reduced default table cannot reveal it. Scores flag suspicious sources, not confirmed attackers. Historical replay is not proof of current attacks.')
    elif any(term in q for term in ('miss','clean','quality','duplicate')):
        largest=df.isna().sum().sort_values(ascending=False).head(5)
        lines.append('Largest observed missing counts: '+', '.join(f'{name}: {value}' for name,value in largest.items())+'. Preview changes in Data Cleaning Studio; do not fit imputation on held-out data.')
    elif any(term in q for term in ('score','model','train','overfit','accuracy')):
        lines.append('Compare development CV first, then report the untouched holdout with support counts. Repeated experimentation can contaminate decisions even without directly refitting on holdout rows. Use fresh capture/host/day validation before deployment.')
    elif any(term in q for term in ('forecast','future','trend')):
        lines.append('Forecast requires a real date field and an ordered holdout. It estimates a scenario from observed history, not future attacker identity or a guaranteed outcome.')
    elif any(term in q for term in ('execute','run code','install','delete','password')):
        lines.append('This local assistant does not execute code, install software, expose credentials or change security controls. Code Lab offers read-only previews.')
    else:
        from cyberlab.workspace import search_workspace
        pages,_=search_workspace(question,df.columns)
        lines.append('Related workflow pages: '+(', '.join(pages[:5]) if pages else 'Overview, Data Quality, ML Lab, Evaluation and Report Generator')+'.')
    lines.append('Local rule-based assistance only; no hosted LLM request or autonomous security action is made.')
    return '\n\n'.join(lines)


def render(page,ctx):
    import streamlit as st
    if page not in PAGES:
        raise ValueError(f'Unknown guidance page: {page}')
    st.header(page)
    if page=='Code Lab':
        st.info('Generate a Python preview for a whitelisted recipe. Nothing is executed in this app; inspect and run independently on a suitable dataset.')
        recipe=st.selectbox('Safe recipe',['Dataset profile','Numeric correlation','Grouped summary'],key='workspace_code_recipe')
        group=st.selectbox('Grouping column',list(ctx['df'].columns),key='workspace_code_group') if recipe=='Grouped summary' else None
        if st.button('Generate code preview',key='workspace_code_generate'):
            try:
                code=code_preview(ctx,recipe,group)
                st.session_state['workspace_code_output']={'fingerprint':ctx['fingerprint'],'recipe':recipe,'group':group,'code':code}
            except ValueError as exc:
                st.error(str(exc))
        result=st.session_state.get('workspace_code_output')
        if result and (result['fingerprint'],result['recipe'],result['group'])==(ctx['fingerprint'],recipe,group):
            st.code(result['code'],language='python')
            st.download_button('Download code preview',result['code'],'analytics_preview.py','text/x-python',key='workspace_code_download')
    elif page=='Domain Review':
        st.info('Record analyst-declared evidence and limits. Saving a review does not authenticate sources, certify readiness or deploy a model.')
        domain=st.selectbox('Review context',['Cybersecurity','General analytics','Business','Education','Regulated / sensitive'],key='workspace_domain_context')
        source=st.text_input('Evidence source / capture reference',value=ctx['source_name'],key='workspace_domain_source')
        observation=st.text_area('Observed finding',key='workspace_domain_observation',max_chars=4000)
        limits=st.text_area('Limitations / alternative explanations',key='workspace_domain_limits',max_chars=4000)
        reviewed=st.checkbox('I reviewed provenance, label meaning and independent validation needs',key='workspace_domain_confirm')
        st.dataframe(pd.DataFrame({'feature':ctx['features'],'dtype':[str(ctx['df'][name].dtype) for name in ctx['features']]}),hide_index=True)
        if st.button('Save session review',key='workspace_domain_save'):
            record={'fingerprint':ctx['fingerprint'],'domain':domain,'source':source,'observation':observation,'limits':limits,'reviewer_declaration':reviewed,'deployment_approval':False}
            st.session_state['workspace_domain_review']=record
        record=st.session_state.get('workspace_domain_review')
        if record and record['fingerprint']==ctx['fingerprint']:
            st.json(record)
            st.download_button('Download domain review',json.dumps(record,indent=2),'domain_review.json','application/json',key='workspace_domain_download')
    else:
        st.caption('Data-aware local assistant, not a hosted generative model. Answers are based on the active table and matching session evidence only.')
        question=st.text_area('Ask about this dataset or workflow',value='What should I check before training?',max_chars=1000,key='workspace_assistant_question')
        if st.button('Answer from active evidence',key='workspace_assistant_answer'):
            try:
                answer=assistant_answer(ctx,question,st.session_state.get('workspace_model_result'))
                st.session_state['workspace_assistant_output']={'fingerprint':ctx['fingerprint'],'question':question,'answer':answer}
            except ValueError as exc:
                st.error(str(exc))
        output=st.session_state.get('workspace_assistant_output')
        if output and output['fingerprint']==ctx['fingerprint'] and output['question']==question:
            st.markdown(output['answer'])
            st.download_button('Download assistant answer',output['answer'],'assistant_answer.md','text/markdown',key='workspace_assistant_download')
