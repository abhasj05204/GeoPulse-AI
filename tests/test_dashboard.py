import requests
from dashboard.api_client import api_base_url,get_json,post_json

class FakeResponse:
    def raise_for_status(self): pass
    def json(self): return {"status":"ok"}

def test_api_base_url_normalizes_render_hostport():
    assert api_base_url("geopulse-api:10000")=="http://geopulse-api:10000"
    assert api_base_url("https://geopulse-api.onrender.com")=="https://geopulse-api.onrender.com"

def test_dashboard_get_client(monkeypatch):
    captured={}
    def fake_get(url,timeout):
        captured.update(url=url,timeout=timeout); return FakeResponse()
    monkeypatch.setattr("dashboard.api_client.requests.get",fake_get)
    assert get_json("http://api:8000","/api/health")["status"]=="ok"
    assert captured["url"]=="http://api:8000/api/health"

def test_dashboard_post_client(monkeypatch):
    captured={}
    def fake_post(url,json,timeout):
        captured.update(url=url,payload=json,timeout=timeout); return FakeResponse()
    monkeypatch.setattr("dashboard.api_client.requests.post",fake_post)
    assert post_json("http://api:8000","/api/predict",{"features":{}})["status"]=="ok"
    assert captured["payload"]=={"features":{}}
