"""Light-theme and navigation presentation acceptance criteria."""
import importlib
from pathlib import Path
import tomllib

ROOT=Path(__file__).resolve().parents[1]


def test_native_streamlit_theme_is_white_with_dark_text():
    theme=tomllib.loads((ROOT/'.streamlit/config.toml').read_text())['theme']
    assert theme['base']=='light'
    assert theme['backgroundColor'].lower()=='#ffffff'
    assert theme['textColor'].lower()=='#0f172a'


def test_shared_chart_theme_uses_white_surfaces_and_visible_labels():
    assert (ROOT/'cyberlab/ui_theme.py').exists(),'Shared light theme is missing'
    module=importlib.import_module('cyberlab.ui_theme')
    import plotly.graph_objects as go
    fig=module.style_chart(go.Figure(go.Bar(x=['A'],y=[1])))
    assert fig.layout.paper_bgcolor=='#ffffff'
    assert fig.layout.plot_bgcolor=='#ffffff'
    assert fig.layout.font.color=='#0f172a'
    assert 'Workspace page' in module.APP_CSS
    assert 'overflow-y' in module.APP_CSS


def test_benchmark_has_collapsible_evidence_instead_of_a_large_warning_wall():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=30).run()
    assert not app.exception and not app.error
    assert any(x.label=='Dataset scope & validation limits' for x in app.expander)
    assert app.sidebar.radio[0].options==['Executive summary','Model comparison','Model diagnostics','Learning curve','Dataset & governance','Report generator','SOC dashboard']
