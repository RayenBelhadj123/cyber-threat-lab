import importlib.util
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import pytest

ROOT=Path(__file__).resolve().parents[1]


def module():
    path=ROOT/'scripts/download_attack_extension.py'
    assert path.exists(), 'Attack extension downloader is missing'
    spec=importlib.util.spec_from_file_location('attack_extension',path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_verified_download_is_atomic_and_refuses_changed_content(tmp_path, monkeypatch):
    mod=module()
    content=b'Source IP,Destination IP,Timestamp,Label\n192.0.2.1,192.0.2.2,2018-12-01 09:00:00,BENIGN\n'
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(content)
        def log_message(self,*args): pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    worker=threading.Thread(target=server.serve_forever,daemon=True)
    worker.start()
    try:
        url=f'http://127.0.0.1:{server.server_port}/file.csv'
        target=tmp_path/'target.csv'
        with pytest.raises(ValueError,match='checksum'):
            mod.download_file(url,target,len(content),'0'*64)
        assert not target.exists()
        assert not target.with_suffix('.csv.part').exists()
        digest=hashlib.sha256(content).hexdigest()
        original_get=mod.requests.get
        attempts=[]
        def flaky_get(*args, **kwargs):
            attempts.append(1)
            if len(attempts)==1:
                raise mod.requests.ConnectionError('Transient interrupted download')
            return original_get(*args, **kwargs)
        monkeypatch.setattr(mod.requests,'get',flaky_get)
        mod.download_file(url,target,len(content),digest)
        assert len(attempts)==2
        assert target.read_bytes()==content
        with pytest.raises(ValueError,match='existing'):
            mod.download_file(url,target,len(content),'0'*64)
        assert target.read_bytes()==content
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_pinned_extension_is_independent_and_bounded():
    mod=module()
    assert len(mod.FILES)==3
    assert {x['remote_path'] for x in mod.FILES}=={'01-12/DrDoS_NTP.csv','01-12/UDPLag.csv','03-11/Portmap.csv'}
    assert all(len(x['sha256'])==64 and x['bytes']>0 for x in mod.FILES)
    assert sum(x['bytes'] for x in mod.FILES)<1_000_000_000
    assert 'CICDDoS2019' in mod.BASE_URL


def test_download_resumes_a_verified_prefix_using_http_range(tmp_path):
    mod=module()
    content=b'first-half-second-half'
    ranges=[]
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start=int(self.headers.get('Range','bytes=0-').split('=')[1].split('-')[0])
            ranges.append(start)
            self.send_response(206)
            self.send_header('Content-Range',f'bytes {start}-{len(content)-1}/{len(content)}')
            self.send_header('Content-Length',str(len(content)-start))
            self.end_headers()
            self.wfile.write(content[start:])
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    worker=threading.Thread(target=server.serve_forever,daemon=True)
    worker.start()
    target=tmp_path/'resume.csv'
    target.with_suffix('.csv.part').write_bytes(content[:10])
    try:
        mod.download_file(f'http://127.0.0.1:{server.server_port}/data',target,len(content),hashlib.sha256(content).hexdigest())
        assert ranges==[10]
        assert target.read_bytes()==content
    finally:
        server.shutdown();server.server_close();worker.join()
