"""DataIQ-inspired, source-preserving workspace data tools.

Only explicit Apply actions publish a new in-memory working copy. All analytics
and exports use the supplied dataset; no demonstration records are generated.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PAGES = ('Project Manager', 'Smart Auto-Setup', 'Data Cleaning Studio', 'Overview',
         'Explore', 'Data Quality', 'Compare', 'Data Dictionary')


def dataset_profile(df):
    return {'rows': len(df), 'columns': len(df.columns),
            'missing_cells': int(df.isna().sum().sum()),
            'duplicate_rows': int(df.duplicated().sum()),
            'empty_columns': int(df.isna().all().sum())}


def data_dictionary(df, date_col=None, target=None, features=None):
    rows = []
    for col in df:
        s = df[col]
        role = 'Target' if col == target else 'Date' if col == date_col else 'Feature' if col in (features or []) else 'Other'
        numeric = pd.api.types.is_numeric_dtype(s)
        example = f'{s.min()} … {s.max()}' if numeric and s.notna().any() else ', '.join(s.dropna().astype(str).head(3))
        rows.append({'column': col, 'role': role, 'dtype': str(s.dtype),
                     'missing': int(s.isna().sum()), 'missing_pct': float(s.isna().mean() * 100) if len(s) else 0.,
                     'unique': int(s.nunique()), 'example_or_range': example})
    return pd.DataFrame(rows, columns=['column', 'role', 'dtype', 'missing', 'missing_pct', 'unique', 'example_or_range'])


def clean_dataframe(df, options, target=None):
    """Preview/apply pure pipeline; audit counts actual changes, not attempts.

    Correlation uses one strongest observed predictor with least-squares fit;
    training excludes masked outliers and the supervised target. Falls back to
    the observed median when insufficient pairs exist. Not a training pipeline:
    imputation for evaluation must be fitted inside each training fold.
    """
    missing_strategy = options.get('missing_strategy', 'none')
    outlier_strategy = options.get('outlier_strategy', 'none')
    if missing_strategy not in ('none', 'mean', 'median', 'correlation') or outlier_strategy not in ('none', 'cap', 'mean', 'median', 'correlation', 'remove'):
        raise ValueError('Unknown cleaning strategy')
    factor = float(options.get('iqr_factor', 1.5))
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError('IQR factor must be positive')
    out, audit = df.copy(deep=True), []
    drops = set(options.get('drop_columns', []))
    if options.get('drop_empty'):
        drops.update(out.columns[out.isna().all()])
    target_actions = target in drops or (missing_strategy != 'none' and target in options.get('missing_columns', [])) or target in options.get('categorical_columns', []) or (outlier_strategy != 'none' and target in options.get('outlier_columns', []))
    if target and target_actions and not options.get('allow_target', False):
        raise ValueError('Cleaning target requires explicit target confirmation')

    def record(action, changed, col=None, detail=''):
        audit.append({'action': action, 'column': col, 'changed': int(changed),
                      'detail': detail, 'target_affected': bool(target and col == target)})

    for col in df:
        if col in drops:
            out = out.drop(columns=col)
            record('Drop column', 1, col)
    if options.get('deduplicate'):
        before = len(out)
        out = out.drop_duplicates().copy()
        record('Deduplicate', before - len(out))
    # Positional unique index keeps masks safe even with duplicate source indices.
    out = out.reset_index(drop=True)

    def repair(col, mask, strategy):
        observed = out.loc[~mask, col].replace([np.inf, -np.inf], np.nan).dropna()
        if observed.empty:
            return 0, 'Unchanged: no observed finite values'
        fallback = observed.mean() if strategy == 'mean' else observed.median()
        estimates = pd.Series(fallback, index=out.index[mask], dtype=float)
        detail = strategy if strategy != 'correlation' else 'Fallback median'
        if strategy == 'correlation':
            candidates = [c for c in smart_setup(out)['features'] if c != col and c != target]
            numeric = out[[col] + candidates].replace([np.inf, -np.inf], np.nan).copy()
            numeric.loc[mask, col] = np.nan
            corr = numeric.corr(numeric_only=True)[col].drop(col).dropna().abs().sort_values(ascending=False)
            if not corr.empty:
                predictor = corr.index[0]
                train = numeric.loc[~mask, [predictor, col]].dropna()
                valid = mask & numeric[predictor].notna()
                if len(train) >= 3 and train[predictor].nunique() > 1:
                    slope, intercept = np.polyfit(train[predictor], train[col], 1)
                    predictions = numeric.loc[valid, predictor] * slope + intercept
                    estimates.loc[predictions.index] = predictions.where(np.isfinite(predictions), fallback)
                    detail = 'Correlation estimate via ' + str(predictor) + '; fallback median for missing predictor'
        out[col] = out[col].astype(float)
        before = out.loc[mask, col].copy()
        out.loc[mask, col] = estimates
        return int((~(before.eq(estimates) | (before.isna() & estimates.isna()))).sum()), detail

    if missing_strategy != 'none':
        for col in options.get('missing_columns', []):
            if col in out and pd.api.types.is_numeric_dtype(out[col]):
                mask = out[col].isna() | ~np.isfinite(out[col])
                changed, detail = repair(col, mask, missing_strategy)
                record('Fill numeric', changed, col, detail)
    for col in options.get('categorical_columns', []):
        if col in out and not pd.api.types.is_numeric_dtype(out[col]):
            mode = out[col].mode(dropna=True)
            count = int(out[col].isna().sum())
            if len(mode):
                out[col] = out[col].fillna(mode.iloc[0])
                record('Fill categorical mode', count, col)
            else:
                record('Fill categorical mode', 0, col, 'Unchanged: no observed mode')
    remove = pd.Series(False, index=out.index)
    if outlier_strategy != 'none':
        for col in options.get('outlier_columns', []):
            if col not in out or not pd.api.types.is_numeric_dtype(out[col]):
                continue
            finite = out[col].replace([np.inf, -np.inf], np.nan)
            q1, q3 = finite.quantile(.25), finite.quantile(.75)
            width = q3 - q1
            lo, hi = q1 - factor * width, q3 + factor * width
            mask = finite.notna() & ~finite.between(lo, hi) if pd.notna(width) and width > 0 else pd.Series(False, index=out.index)
            if outlier_strategy == 'remove':
                remove |= mask
                continue
            if outlier_strategy == 'cap':
                out[col] = out[col].astype(float)
                out.loc[mask, col] = finite.loc[mask].clip(lo, hi)
                changed, detail = int(mask.sum()), f'IQR bounds {lo:g}, {hi:g}'
            else:
                changed, detail = repair(col, mask, outlier_strategy)
            record('Repair outliers', changed, col, detail)
        if outlier_strategy == 'remove':
            out = out.loc[~remove].copy()
            record('Remove outlier rows', int(remove.sum()))
    return out.reset_index(drop=True), audit


def filter_rows(df, category=None, values=None, numeric=None, lower=None, upper=None):
    mask = pd.Series(True, index=df.index)
    if category in df and values is not None:
        mask &= df[category].isin(values)
    if numeric in df and pd.api.types.is_numeric_dtype(df[numeric]):
        if lower is not None:
            mask &= df[numeric].ge(lower)
        if upper is not None:
            mask &= df[numeric].le(upper)
    return df.loc[mask].copy()


def numeric_correlation(df, columns=None, method='pearson'):
    numeric = df.select_dtypes(include='number')
    if columns is not None:
        numeric = numeric[[c for c in columns if c in numeric]]
    return numeric.replace([np.inf, -np.inf], np.nan).corr(method=method)


def compare_groups(df, group, metrics, aggregation='mean'):
    metrics = [c for c in metrics if c in df and c != group and pd.api.types.is_numeric_dtype(df[c])]
    if group not in df or not metrics:
        return pd.DataFrame()
    if aggregation not in ('mean', 'median', 'sum', 'std', 'count', 'min', 'max'):
        raise ValueError('Unsupported aggregation')
    data = df[[group] + metrics].copy()
    data[group] = data[group].astype('string').fillna('(missing)')
    result = data.groupby(group, observed=True)[metrics].agg(aggregation)
    group_name = 'group'
    while group_name in metrics:
        group_name = '_' + group_name
    result.index.name = group_name
    return result.reset_index()


def _export_table(data, label, key, name):
    st.download_button(label, data.to_csv(index=False).encode('utf-8'), name, 'text/csv', key=key)


def render(page, ctx):
    """Render one owned route. Parent owns dataset/configuration sidebar."""
    if page not in PAGES:
        raise ValueError(f'Unknown workspace data page: {page}')
    st.subheader(page)
    df = ctx.get('df')
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.info('No rows available. Load a non-empty dataset using the sidebar.')
        return
    if not df.columns.is_unique or any(not isinstance(c, str) for c in df.columns):
        st.error('Column names must be unique strings. Correct the input file before analysis.')
        return
    raw = ctx.get('raw_df')
    raw = raw if isinstance(raw, pd.DataFrame) and raw.columns.is_unique else df
    target = ctx.get('target') if ctx.get('target') in df else None
    date = ctx.get('date_col') if ctx.get('date_col') in df else None
    numeric = df.select_dtypes(include='number').columns.tolist()
    prefix = 'workspace_data_' + page.lower().replace(' ', '_').replace('-', '_') + '_'
    if page == 'Project Manager':
        st.markdown(f"**Project:** {ctx.get('project_name', 'Untitled')}  \n**Source:** {ctx.get('source_name', 'Unknown')}")
        manifest = {key: ctx.get(key) for key in ('project_name', 'source_name', 'mode', 'fingerprint', 'target', 'date_col', 'features')}
        manifest['active_profile'], manifest['raw_profile'] = dataset_profile(df), dataset_profile(raw)
        st.dataframe(pd.DataFrame([manifest['raw_profile'], manifest['active_profile']], index=['Original', 'Working copy']), width='stretch')
        st.caption('Change project name, source, target, date, and features in the sidebar. Original source files are never modified here.')
        st.download_button('Download project manifest', json.dumps(manifest, indent=2, default=str), 'workspace_project.json', 'application/json', key=prefix+'manifest')
        _export_table(df, 'Download working dataset', prefix+'csv', 'workspace_dataset.csv')
        st.dataframe(df.head(100), width='stretch')
    elif page == 'Data Cleaning Studio':
        st.caption('Preview first, then apply to an in-memory working copy. For model evaluation fit imputers on training data only.')
        categories = [c for c in df if c not in numeric and c != date]
        options = {
            'deduplicate': st.checkbox('Remove duplicate rows', key=prefix+'deduplicate'),
            'drop_empty': st.checkbox('Drop fully empty columns', key=prefix+'empty'),
            'drop_columns': st.multiselect('Drop selected columns', df.columns.tolist(), key=prefix+'drop'),
            'missing_columns': st.multiselect('Numeric columns to repair', numeric, key=prefix+'missing'),
            'missing_strategy': st.selectbox('Numeric missing strategy', ['none', 'median', 'mean', 'correlation'], key=prefix+'missing_strategy'),
            'categorical_columns': st.multiselect('Categorical columns to fill with observed mode', categories, key=prefix+'categorical'),
            'outlier_columns': st.multiselect('Outlier columns', numeric, key=prefix+'outliers'),
            'outlier_strategy': st.selectbox('Outlier strategy', ['none', 'cap', 'median', 'mean', 'correlation', 'remove'], key=prefix+'outlier_strategy'),
            'iqr_factor': st.number_input('IQR fence multiplier', min_value=.1, max_value=10., value=1.5, step=.1, key=prefix+'factor'),
        }
        st.warning('Changing target labels, dropping target, or removing rows can alter class balance. Target edits require confirmation; no labels are invented.')
        options['allow_target'] = st.checkbox('I explicitly allow target changes', key=prefix+'allow_target')
        # Include content hash, not merely dataset identity, so stale previews
        # cannot overwrite a newly applied working copy or a different source.
        signature = (ctx.get('fingerprint'), tuple(df.columns), tuple(str(d) for d in df.dtypes),
                     int(pd.util.hash_pandas_object(df, index=True).sum()), json.dumps(options, sort_keys=True))
        preview_key = prefix+'preview'
        if st.button('Preview changes', key=prefix+'preview_button'):
            try:
                cleaned, audit = clean_dataframe(df, options, target)
                st.session_state[preview_key] = (signature, cleaned, audit)
            except ValueError as exc:
                st.session_state.pop(preview_key, None)
                st.error(str(exc))
        preview = st.session_state.get(preview_key)
        current = preview is not None and preview[0] == signature
        if current:
            _, cleaned, audit = preview
            st.markdown('#### Proposed working copy')
            st.dataframe(pd.DataFrame([dataset_profile(df), dataset_profile(cleaned)], index=['Before', 'After']), width='stretch')
            st.dataframe(cleaned.head(100), width='stretch')
            st.dataframe(pd.DataFrame(audit), width='stretch')
            _export_table(cleaned, 'Download preview CSV', prefix+'preview_csv', 'workspace_preview.csv')
        elif preview is not None:
            st.info('Dataset or options changed. Generate a fresh preview before applying.')
        if st.button('Apply preview', key=prefix+'apply', disabled=not current):
            st.session_state['workspace_cleaned'] = preview[1].copy(deep=True)
            st.session_state['workspace_cleaning_audit'] = list(ctx.get('cleaning_audit') or []) + preview[2]
            st.session_state.pop(preview_key, None)
            st.success('Working copy updated. Source files are unchanged.')
            st.rerun()
    elif page == 'Explore':
        categories = [c for c in df if c != date and df[c].nunique() <= 100 and (c not in numeric or c == target or df[c].nunique() <= 20)]
        category = st.selectbox('Category filter', [None] + categories, format_func=lambda x: 'No category filter' if x is None else x, key=prefix+'category')
        values = None
        if category:
            choices = sorted(df[category].dropna().unique().tolist(), key=str)
            values = st.multiselect('Keep category values (empty selects no rows)', choices, default=choices, key=prefix+'values')
        range_col = st.selectbox('Numeric range filter', [None] + numeric, format_func=lambda x: 'No range filter' if x is None else x, key=prefix+'range')
        lower = upper = None
        if range_col:
            finite = df[range_col].replace([np.inf, -np.inf], np.nan).dropna()
            if not finite.empty:
                left, right = st.columns(2)
                lower = left.number_input('Minimum inclusive', value=float(finite.min()), key=prefix+'min')
                upper = right.number_input('Maximum inclusive', value=float(finite.max()), key=prefix+'max')
        view = filter_rows(df, category, values, range_col, lower, upper)
        st.caption(f'{len(view):,} of {len(df):,} rows selected. Exports contain the full filtered rows; display table is limited to 200.')
        st.dataframe(view.head(200), width='stretch')
        _export_table(view, 'Download filtered rows', prefix+'csv', 'workspace_explore.csv')
        if view.empty:
            st.info('No rows match these filters.')
            return
        if not numeric:
            st.info('No numeric columns; inspect category frequencies below.')
            col = st.selectbox('Category to count', df.columns.tolist(), key=prefix+'count_col')
            counts = view[col].astype('string').fillna('(missing)').value_counts().head(50).rename_axis('value').reset_index(name='rows')
            st.plotly_chart(px.bar(counts, x='value', y='rows', title=f'{col}: top observed values'), width='stretch', key=prefix+'counts')
            return
        chart = st.selectbox('Chart type', ['Histogram', 'Scatter', 'Line', 'Box', 'Violin'], key=prefix+'chart')
        metric = st.selectbox('Metric', numeric, key=prefix+'metric')
        color = st.selectbox('Color / group', [None] + categories, format_func=lambda x: 'No grouping' if x is None else x, key=prefix+'color')
        display = view.copy()
        if color:
            display[color] = display[color].astype('string').fillna('(missing)')
        if chart == 'Histogram':
            bins = st.slider('Histogram bins', 5, 100, 30, key=prefix+'bins')
            fig = px.histogram(display, x=metric, color=color, nbins=bins, marginal='box', title=f'{metric}: distribution')
        elif chart == 'Scatter':
            other = st.selectbox('Y metric', numeric, index=min(1, len(numeric)-1), key=prefix+'y')
            # Bound only individual-point plots; never silently sample aggregates.
            if len(display) > 5000:
                display = display.iloc[np.linspace(0, len(display)-1, 5000, dtype=int)]
                st.caption('Scatter plot uses 5,000 evenly spaced source rows; exported data is unsampled.')
            fig = px.scatter(display, x=metric, y=other, color=color, opacity=.65)
        elif chart == 'Line':
            if not date:
                st.info('Select a date column in the sidebar for a time trend.')
                return
            trend = pd.DataFrame({'timestamp': pd.to_datetime(view[date], errors='coerce', format='mixed', utc=True), 'value': view[metric]})
            trend = trend.dropna().groupby('timestamp', as_index=False)['value'].mean().sort_values('timestamp')
            window = st.slider('Rolling observations', 1, 100, 5, key=prefix+'rolling')
            trend['rolling_mean'] = trend['value'].rolling(window, min_periods=1).mean()
            if trend.empty:
                st.info('No parseable dates with numeric observations.')
                return
            fig = px.line(trend, x='timestamp', y=['value', 'rolling_mean'], title=f'{metric}: mean by timestamp and rolling mean')
        elif chart == 'Box':
            fig = px.box(display, x=color, y=metric, color=color, points='outliers')
        else:
            fig = px.violin(display, x=color, y=metric, color=color, box=True, points=False)
        st.plotly_chart(fig, width='stretch', key=prefix+'chart_output')
    elif page == 'Compare':
        if not numeric:
            st.info('No numeric measures available for grouped comparison.')
            st.dataframe(df.head(100), width='stretch')
            _export_table(df, 'Download current dataset', prefix+'csv', 'workspace_compare.csv')
            return
        groups = [c for c in df if c == date or c == target or df[c].nunique() <= 100]
        if not groups:
            st.info('No low-cardinality group or date selected. Select a date column in the sidebar.')
            return
        default = groups.index('time_group') if 'time_group' in groups else 0
        group = st.selectbox('Group by', groups, index=default, key=prefix+'group')
        metrics = st.multiselect('Comparison metrics', numeric, default=[c for c in ctx.get('features', numeric[:3]) if c in numeric][:6], key=prefix+'metrics')
        aggregation = st.selectbox('Aggregation', ['mean', 'median', 'std', 'sum', 'count', 'min', 'max'], key=prefix+'aggregation')
        data = df.copy()
        if group == date:
            frequency = st.selectbox('Calendar interval', ['D', 'W', 'M', 'Y'], key=prefix+'frequency')
            parsed = pd.to_datetime(data[group], errors='coerce', format='mixed', utc=True)
            data[group] = parsed.dt.tz_localize(None).dt.to_period(frequency).astype('string').mask(parsed.isna(), pd.NA)
        result = compare_groups(data, group, metrics, aggregation)
        if result.empty:
            st.info('Select at least one numeric metric other than the grouping column.')
            return
        st.dataframe(result, width='stretch', hide_index=True)
        _export_table(result, 'Download grouped comparison', prefix+'csv', 'workspace_compare.csv')
        # A fresh long-form frame avoids collisions with source columns named
        # group, metric, or value, without renaming any exported source metric.
        long = pd.DataFrame([{'group': row.iloc[0], 'metric': col, 'value': row[col]}
                             for _, row in result.iterrows() for col in result.columns[1:]])
        st.plotly_chart(px.bar(long, x='group', y='value', color='metric', barmode='group', title=f'{aggregation.title()} by {group}'), width='stretch', key=prefix+'bar')
        counts = data[group].astype('string').fillna('(missing)').value_counts().rename_axis('group').reset_index(name='rows')
        st.dataframe(counts, width='stretch', hide_index=True)
        st.caption('Group counts are shown to expose unequal support. Aggregates are descriptive, not causal or significance tests.')
    elif page == 'Overview':
        profile = dataset_profile(df)
        cards = st.columns(4)
        for card, (label, value) in zip(cards, list(profile.items())[:4]):
            card.metric(label.replace('_', ' ').title(), f'{value:,}')
        if target:
            counts = df[target].astype('string').fillna('(missing)').value_counts().rename_axis('label').reset_index(name='rows')
            if df[target].nunique() <= 30:
                st.plotly_chart(px.bar(counts, x='label', y='rows', title='Observed target composition'), width='stretch', key=prefix+'target')
            st.dataframe(df[target].describe().to_frame('value'), width='stretch')
        else:
            st.info('No target selected. General dataset profiling remains available.')
        if numeric:
            metric = st.selectbox('Overview metric', numeric, index=numeric.index(target) if target in numeric else 0, key=prefix+'metric')
            if date:
                trend = pd.DataFrame({'timestamp': pd.to_datetime(df[date], errors='coerce', format='mixed', utc=True), 'value': df[metric]})
                trend = trend.dropna().groupby('timestamp', as_index=False)['value'].mean().sort_values('timestamp')
                if not trend.empty:
                    st.plotly_chart(px.line(trend, x='timestamp', y='value', title=f'{metric}: mean by timestamp'), width='stretch', key=prefix+'trend')
            defaults = list(dict.fromkeys(([target] if target in numeric else []) + [c for c in ctx.get('features', []) if c in numeric]))[:12]
            if not defaults:
                defaults = numeric[:12]
            cols = st.multiselect('Correlation columns', numeric, default=defaults, key=prefix+'corr_cols')
            method = st.selectbox('Correlation method', ['pearson', 'spearman'], key=prefix+'method')
            corr = numeric_correlation(df, cols, method)
            if len(corr) >= 2:
                st.plotly_chart(px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale='RdBu_r', title='Numeric relationships'), width='stretch', key=prefix+'corr')
                st.dataframe(corr, width='stretch')
                st.caption('Correlation is association, not causation. Default columns use the sidebar feature selection; label encoding is never fabricated.')
            st.dataframe(df[numeric].describe().T, width='stretch')
        else:
            st.info('No numeric columns. Review the source types before numeric analysis.')
        st.dataframe(df.head(100), width='stretch')
        _export_table(df, 'Download overview dataset', prefix+'csv', 'workspace_overview.csv')
    elif page == 'Data Quality':
        profile = dataset_profile(df)
        before = dataset_profile(raw)
        st.dataframe(pd.DataFrame({'metric': list(profile), 'original': list(before.values()), 'working': list(profile.values()),
                                   'change': [profile[k] - before[k] for k in profile]}), width='stretch', hide_index=True)
        report = data_dictionary(df, date, target, ctx.get('features'))
        report['infinite'] = [int(np.isinf(df[c]).sum()) if c in numeric else 0 for c in report['column']]
        report['constant'] = report['unique'].le(1)
        only_issues = st.checkbox('Only columns needing attention', key=prefix+'issues')
        view = report.loc[(report['missing'] > 0) | (report['infinite'] > 0) | report['constant']] if only_issues else report
        st.dataframe(view, width='stretch', hide_index=True)
        st.plotly_chart(px.bar(report, x='column', y='missing_pct', title='Missing cells by column (%)'), width='stretch', key=prefix+'missing_chart')
        st.markdown('#### Applied cleaning audit')
        audit = pd.DataFrame(ctx.get('cleaning_audit') or [])
        st.dataframe(audit, width='stretch')
        if audit.empty:
            st.caption('No user cleaning operations have been applied. Checks above describe current data, not claimed repairs.')
        _export_table(report, 'Download quality report', prefix+'report', 'workspace_quality.csv')
        _export_table(df, 'Download cleaned dataset CSV', prefix+'cleaned', 'workspace_cleaned.csv')
        if not audit.empty:
            _export_table(audit, 'Download cleaning audit', prefix+'audit', 'workspace_cleaning_audit.csv')
    elif page == 'Data Dictionary':
        dictionary = data_dictionary(df, date, target, ctx.get('features'))
        search = st.text_input('Search column names', key=prefix+'search')
        roles = st.multiselect('Column roles', dictionary['role'].unique().tolist(), key=prefix+'roles')
        view = dictionary[dictionary['column'].str.contains(search, case=False, regex=False)]
        if roles:
            view = view[view['role'].isin(roles)]
        st.dataframe(view, width='stretch', hide_index=True)
        st.plotly_chart(px.bar(dictionary.groupby('dtype').size().reset_index(name='columns'), x='dtype', y='columns', title='Schema composition'), width='stretch', key=prefix+'schema')
        st.caption('Descriptions are observed types, examples, and selected roles; unprovided domain definitions are not invented.')
        _export_table(view, 'Download filtered dictionary', prefix+'csv', 'workspace_dictionary.csv')
    elif page == 'Smart Auto-Setup':
        setup = smart_setup(df)
        st.markdown(f"**Suggested task:** {setup['task']}  \n**Target:** {setup['target'] or 'Not detected'}  \n**Date:** {setup['date_col'] or 'Not detected'}")
        st.caption('Recommendations only: review and select settings in the sidebar. Labels are read from the dataset, never generated.')
        st.dataframe(pd.DataFrame({'recommended_feature': setup['features']}), width='stretch')
        st.dataframe(pd.DataFrame(setup['date_candidates'], columns=['column', 'parsed_pct']), width='stretch')
        st.write('Target candidates:', setup['target_candidates'])
        if not setup['features']:
            st.warning('No eligible numeric features. Review input types and missing values.')
        st.download_button('Download setup recommendations', json.dumps(setup, indent=2), 'workspace_setup.json', 'application/json', key=prefix+'json')


def smart_setup(df):
    dates, targets = [], []
    numeric = df.select_dtypes(include='number').columns.tolist()
    for col in df:
        s, name = df[col], str(col).lower()
        named_date = any(token in name for token in ('timestamp', 'datetime', 'date', 'time')) and name != 'time_group'
        if (named_date and col not in numeric) or pd.api.types.is_datetime64_any_dtype(s):
            ratio = pd.to_datetime(s, errors='coerce', format='mixed', utc=True).notna().mean() if len(s) else 0.
            if ratio >= .5:
                dates.append({'column': col, 'parsed_pct': float(ratio * 100)})
        if s.nunique() > 1 and any(token in name for token in ('malicious', 'target', 'label', 'outcome', 'attack', 'class')) and not any(token in name for token in ('copy', 'pred', 'prob', 'score', 'lag')):
            targets.append(col)
    target = targets[0] if targets else None
    date = dates[0]['column'] if dates else None
    features = []
    for col in numeric:
        s, name = df[col], str(col).lower()
        if col == target or col == date or name in ('time_group', 'id', 'index') or name.endswith('_id'):
            continue
        if any(token in name for token in ('label', 'malicious', 'attack', 'target', 'predicted', 'prediction', 'probability', 'leak')):
            continue
        if target and (str(target).lower() in name or s.equals(df[target])):
            continue
        if s.nunique() > 1 and s.isna().mean() <= .6:
            features.append(col)
    task = 'Not ready' if target is None else 'Classification' if target not in numeric or df[target].nunique() <= 20 else 'Regression'
    return {'target': target, 'date_col': date, 'features': features, 'task': task,
            'date_candidates': dates, 'target_candidates': targets}
