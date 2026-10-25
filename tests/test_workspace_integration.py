from pathlib import Path
import importlib
from streamlit.testing.v1 import AppTest
from cyberlab.workspace import PAGE_NAMES

ROOT=Path(__file__).resolve().parents[1]


def test_every_registered_page_has_exactly_one_owner():
    ownership={page:['core'] for page in ('Start Here','Executive Summary','Report Generator')}
    for name in ('workspace_data','workspace_ml','workspace_advanced','workspace_guidance'):
        path=ROOT/'cyberlab'/f'{name}.py'
        assert path.exists(), f'Missing page owner: {name}'
        for page in importlib.import_module(f'cyberlab.{name}').PAGES:
            ownership.setdefault(page,[]).append(name)
    assert set(ownership)==set(PAGE_NAMES)
    assert all(len(owners)==1 for owners in ownership.values())


def test_all_workspace_routes_render_in_the_actual_entrypoint():
    app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=45).run()
    app.toggle(key='full_analytics_workspace').set_value(True).run()
    for page in PAGE_NAMES:
        app.radio(key='workspace_page').set_value(page).run()
        assert not app.exception, f'{page}: {[x.message for x in app.exception]}'
        assert not app.error, f'{page}: {[x.value for x in app.error]}'
        assert app.title[0].value==page
